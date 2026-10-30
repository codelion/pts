"""
Command-line interface for PTS.

    pts run       search for pivotal events at one or more scales
    pts enrich    add latent meta-token events to an existing dataset
    pts fit-jlens calibrate a Jacobian lens for a model
    pts link      link latent, token, and sentence events into causal chains
    pts migrate   convert legacy pivotal-token / thought-anchor files to PTS events
    pts export    render events into a downstream format
    pts push      upload a dataset to Hugging Face
    pts steer     generate with steering vectors applied
    pts build-vector  build a mean-difference steering vector from paired examples
    pts serve     OpenAI-compatible server with steering (wraps mlx_lm.server)
    pts discover  find pivotal tokens for a behavior and turn them into validated steering vectors
"""

import argparse
import json
import logging
import os
import shutil
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    numeric = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pts.log")],
        force=True,
    )


def _layer_fraction(args) -> tuple:
    lo, hi = args.workspace_layer_fraction
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"--workspace-layer-fraction must satisfy 0 <= lo < hi <= 1, got {lo} {hi}"
        )
    return (lo, hi)


def _workspace_layers(args) -> Optional[List[int]]:
    layers = getattr(args, "workspace_layers", None)
    if not layers or layers == ["auto"]:
        return None
    return [int(l) for l in layers]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_pts(args) -> None:
    setup_logging(args.log_level)

    from tqdm import tqdm

    from .dataset import create_oracle_from_dataset, load_dataset
    from .event_storage import EventStorage
    from .linking import EventLinker

    granularity = args.granularity
    # alias: --generate-thought-anchors is sentence-level PTS.
    if getattr(args, "generate_thought_anchors", False):
        logger.info(
            "--generate-thought-anchors maps to --granularity sentence; "
            "prefer the new flag."
        )
        granularity = "sentence"

    granularities = (
        ["token", "sentence", "latent"] if granularity == "all" else [granularity]
    )

    examples = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        config=args.config,
        sample_size=args.sample_size,
        seed=args.seed,
        query_key=args.query_key,
        answer_key=args.answer_key,
    )
    if not examples:
        logger.error(f"No examples loaded from {args.dataset}")
        sys.exit(1)
    logger.info(f"Loaded {len(examples)} examples from {args.dataset}")

    oracle = None
    if granularities != ["latent"]:
        # Latent PTS is observational -- it never estimates success probability,
        # so it needs no oracle.
        oracle = create_oracle_from_dataset(examples, debug_mode=args.debug)
        if args.skip_embeddings:
            oracle.skip_embeddings = True

    storage = EventStorage(filepath=args.output_path)

    searcher_kwargs = dict(
        device=args.device,
        dtype=args.dtype,
        prob_threshold=args.prob_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        log_level=getattr(logging, args.log_level.upper()),
        debug_mode=args.debug,
    )

    from .searchers.reasoning import ReasoningSearcher

    searcher = ReasoningSearcher(
        model_name=args.model,
        oracle=oracle,
        granularities=granularities,
        event_storage=storage,
        readout_method=args.readout_method,
        jlens_path=args.jlens_path,
        workspace_layers=_workspace_layers(args),
        layer_fraction=_layer_fraction(args),
        window_before=args.window_before,
        link_threshold=args.link_threshold,
        # These four were registered on the parser and then never read, so
        # setting them changed nothing.
        readout_top_k=args.readout_top_k,
        min_score=args.min_score,
        keep_per_position=args.keep_per_position,
        enable_verification=args.enable_verification,
        skip_embeddings=args.skip_embeddings,
        **searcher_kwargs,
    )

    total = 0
    processed = 0
    for i, example in enumerate(
        tqdm(examples[: args.max_examples], desc="Processing examples")
    ):
        query = example["query"]
        if not query.strip():
            logger.warning(f"Skipping empty query in example {i}")
            continue

        try:
            events = searcher.search(
                query=query,
                system_prompt=args.system_prompt,
                task_type=args.task_type,
                dataset_id=args.dataset,
                item_id=example.get("item_id", str(i)),
                max_generations=args.max_generations,
                min_prob=args.min_prob,
                max_prob=args.max_prob,
                category=example.get("metadata", {}).get("category"),
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted; saving what we have")
            break
        except Exception as e:
            # Log the traceback rather than swallowing it. the legacy code caught bare
            # Exception and printed one line, which is how a guaranteed
            # UnboundLocalError in the anchor search went unnoticed.
            logger.exception(f"Error processing example {i}: {e}")
            continue

        if events:
            processed += 1
            total += len(events)
            logger.info(f"Found {len(events)} events for example {i}")

    if total:
        storage.save()
        summary = storage.summary()
        logger.info(f"Found events in {processed}/{min(len(examples), args.max_examples)} examples")
        logger.info(f"Saved {len(storage)} events to {args.output_path}")
        logger.info(f"  latent:   {summary['latent_events']}")
        logger.info(f"  token:    {summary['token_events']}")
        logger.info(f"  sentence: {summary['sentence_events']}")
        logger.info(f"  links:    {summary['num_links']}")
    else:
        logger.info("No events found; nothing saved")


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------

def migrate(args) -> None:
    setup_logging(args.log_level)

    from .event_storage import EventStorage

    storage = EventStorage(filepath=args.input_path)
    if not len(storage):
        logger.error(f"No records loaded from {args.input_path}")
        sys.exit(1)

    storage.save(args.output_path)

    summary = storage.summary()
    logger.info(f"Migrated {len(storage)} records -> {args.output_path}")
    logger.info(f"  token events:    {summary['token_events']}")
    logger.info(f"  sentence events: {summary['sentence_events']}")
    logger.info(f"  latent events:   {summary['latent_events']}")


# ---------------------------------------------------------------------------
# fit-jlens
# ---------------------------------------------------------------------------

def fit_jlens(args) -> None:
    setup_logging(args.log_level)

    from transformers import AutoTokenizer

    from .latent.activations import get_layer_modules, resolve_workspace_layers
    from .latent.jlens import JLens
    from .searchers.base import load_causal_lm, select_device

    device = select_device(args.device)
    logger.info(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Fitting differentiates through the network; fp16 Jacobians underflow badly,
    # so calibration runs in fp32 regardless of what inference would use. There
    # is deliberately no --dtype flag here.
    model = load_causal_lm(args.model, device, purpose="fit_jlens")

    num_layers = len(get_layer_modules(model))
    layers = resolve_workspace_layers(
        num_layers,
        layers=_workspace_layers(args),
        fraction=_layer_fraction(args),
        max_layers=args.max_layers,
    )
    logger.info(f"Fitting J-lens for layers {layers} of {num_layers}")

    texts = _calibration_texts(args)
    logger.info(f"Using {len(texts)} calibration sequences of up to {args.seq_len} tokens")

    jlens = JLens(model, tokenizer)
    jlens.fit(
        texts,
        layers=layers,
        seq_len=args.seq_len,
        basis_chunk=args.basis_chunk,
        device=device,
    )
    jlens.save(args.output_path)

    logger.info(f"Saved J-lens to {args.output_path}")
    logger.info(f"Read it back with: pts enrich --jlens-path {args.output_path} --readout-method jlens")


def _calibration_texts(args) -> List[str]:
    """Calibration text for J-lens fitting.

    The paper averages the Jacobian over pretraining-like text. Any reasonably
    generic corpus works -- the lens is a property of the model, not the task --
    so the source dataset is a fine default and needs no labels.
    """
    if args.calibration_file:
        with open(args.calibration_file) as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts[: args.num_sequences]

    from .dataset import load_dataset

    examples = load_dataset(
        dataset_name=args.calibration_dataset,
        split=args.split,
        sample_size=args.num_sequences,
        seed=args.seed,
    )
    if not examples:
        raise ValueError(
            f"Could not load calibration text from {args.calibration_dataset}. "
            "Pass --calibration-file with one sequence per line instead."
        )
    return [e["query"] for e in examples][: args.num_sequences]


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------

def enrich(args) -> None:
    setup_logging(args.log_level)

    from transformers import AutoTokenizer

    from .event_storage import EventStorage
    from .latent.jlens import load_readout
    from .latent.metatokens import enrich_events_with_latent
    from .linking import EventLinker
    from .searchers.base import load_causal_lm, select_device

    storage = EventStorage(filepath=args.input_path)
    if not len(storage):
        logger.error(f"No events loaded from {args.input_path}")
        sys.exit(1)
    logger.info(f"Loaded {len(storage)} events from {args.input_path}")

    if not args.with_latent:
        logger.error("Nothing to do: pass --with-latent")
        sys.exit(1)

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_causal_lm(args.model, device, dtype=args.dtype)

    readout = load_readout(
        model, tokenizer, method=args.readout_method, jlens_path=args.jlens_path
    )

    events = enrich_events_with_latent(
        list(storage),
        model,
        tokenizer,
        readout,
        layers=_workspace_layers(args),
        layer_fraction=_layer_fraction(args),
        window_before=args.window_before,
        top_k=args.readout_top_k,
        min_score=args.min_score,
        keep_per_position=args.keep_per_position,
        allow_model_mismatch=args.allow_model_mismatch,
        latent_model_id=args.model,
    )

    linker = EventLinker(threshold=args.link_threshold, window=args.window_before)
    links = linker.link(events)

    out = EventStorage()
    out.add_events(events)
    out.save(args.output_path)

    summary = out.summary()
    logger.info(f"Wrote {len(out)} events to {args.output_path}")
    logger.info(f"  latent events added: {summary['latent_events']}")
    logger.info(f"  links:               {len(links)}")

    if args.shuffle_control:
        control = linker.shuffle_control(events)
        logger.info("Shuffle control (link scores against mismatched queries):")
        logger.info(f"  observed mean: {control['observed_mean']:.4f} (n={control['observed_n']})")
        logger.info(f"  shuffled mean: {control['shuffled_mean']:.4f} (n={control['shuffled_n']})")
        if control["observed_mean"] <= control["shuffled_mean"]:
            logger.warning(
                "Observed link scores are NOT above the shuffled baseline. The "
                "latent-precedes-emitted structure in this dataset is not "
                "distinguishable from chance."
            )


# ---------------------------------------------------------------------------
# link
# ---------------------------------------------------------------------------

def link(args) -> None:
    setup_logging(args.log_level)

    from .event_storage import EventStorage
    from .linking import EventLinker

    storage = EventStorage(filepath=args.input_path)
    if not len(storage):
        logger.error(f"No events loaded from {args.input_path}")
        sys.exit(1)

    linker = EventLinker(threshold=args.link_threshold, window=args.window_before)
    events = list(storage)
    links = linker.link(events)

    out = EventStorage()
    out.add_events(events)
    out.save(args.output_path)
    logger.info(f"Wrote {len(out)} events with {len(links)} links to {args.output_path}")

    if args.edges_output_path:
        with open(args.edges_output_path, "w") as f:
            for l in links:
                f.write(json.dumps(l.to_dict()) + "\n")
        logger.info(f"Wrote {len(links)} edges to {args.edges_output_path}")

    if args.shuffle_control:
        control = linker.shuffle_control(events)
        logger.info("Shuffle control:")
        logger.info(f"  observed mean: {control['observed_mean']:.4f} (n={control['observed_n']})")
        logger.info(f"  shuffled mean: {control['shuffled_mean']:.4f} (n={control['shuffled_n']})")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export(args) -> None:
    setup_logging(args.log_level)

    from .event_storage import EventStorage
    from .exporters import EventExporter

    storage = EventStorage(filepath=args.input_path)
    if not len(storage):
        logger.error(f"No events loaded from {args.input_path}")
        sys.exit(1)

    exporter = EventExporter(storage)

    if args.format == "causal_events":
        exporter.export_causal_events(
            args.output_path,
            min_prob_delta=args.min_prob_delta,
            min_score=args.min_score,
        )
    elif args.format == "metatokens":
        exporter.export_metatokens(args.output_path, min_score=args.min_score)
    elif args.format == "pivotal_tokens":
        exporter.export_pivotal_tokens(args.output_path, min_prob_delta=args.min_prob_delta)
    elif args.format == "thought_anchors":
        exporter.export_thought_anchors(args.output_path, min_prob_delta=args.min_prob_delta)
    elif args.format == "dpo":
        exporter.export_dpo(
            args.output_path,
            model_name=args.model,
            dataset=args.dataset,
            split=args.split,
            find_rejected_tokens=args.find_rejected_tokens,
            num_candidates=args.num_candidates,
            min_prob_delta=args.min_prob_delta,
            max_pairs=args.max_pairs,
            balance=args.balance,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.format == "steering":
        exporter.export_steering_vectors(
            args.output_path,
            model_name=args.model,
            layer_nums=args.layer_nums,
            select_layer=args.select_layer,
            num_clusters=args.num_clusters,
            pca_components=args.pca_components,
            batch_size=args.batch_size,
            min_prob_delta=args.min_prob_delta,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        logger.error(f"Unsupported export format: {args.format}")
        sys.exit(1)

    if args.hf_push:
        if not args.hf_repo_id:
            logger.error("--hf-push needs --hf-repo-id")
            sys.exit(1)
        _push(args.output_path, args.hf_repo_id, args.private, args.model, args.format)


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

def push_to_hf(args) -> None:
    setup_logging(args.log_level)
    _push(
        args.input_path,
        args.hf_repo_id,
        args.private,
        args.model,
        getattr(args, "format", None),
        no_readme=args.no_readme,
        example_prompt=getattr(args, "example_prompt", None),
    )


def _push(
    path: str,
    repo_id: str,
    private: bool,
    model_name: Optional[str],
    file_type: Optional[str] = None,
    no_readme: bool = False,
    example_prompt: Optional[str] = None,
) -> None:
    from huggingface_hub import create_repo, upload_file

    from .exporters import detect_file_type, generate_dataset_card

    if os.path.isdir(path):
        _push_discovery(path, repo_id, private, no_readme, example_prompt=example_prompt)
        return

    file_type = file_type or detect_file_type(path)

    create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True)
    upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Uploaded {path} to {repo_id}")

    if no_readme:
        return

    card = generate_dataset_card(path, file_type=file_type, model_name=model_name)
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(card)
        card_path = f.name
    try:
        upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info(f"Wrote dataset card for {repo_id}")
    finally:
        os.unlink(card_path)


def _push_discovery(output_dir: str, repo_id: str, private: bool, no_readme: bool = False,
                    method: Optional[str] = None, example_prompt: Optional[str] = None) -> None:
    """Publish a ``pts discover`` run as a model repo: ``increase/`` and
    ``decrease/`` vectors, ``report.json``, and a card generated from it."""
    import json
    import tempfile

    from huggingface_hub import create_repo, upload_folder

    from .steering.card import DIRECTIONS, MAIN, discovery_card, pick_examples

    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        raise SystemExit(f"{output_dir} has no report.json; push a finished `pts discover` output directory")
    with open(report_path) as f:
        report = json.load(f)
    method = method or MAIN
    with tempfile.TemporaryDirectory() as stage:
        published = []
        for d in DIRECTIONS:
            src = os.path.join(output_dir, f"{method}_{d}")
            if (report["methods"].get(method) or {}).get(d) and os.path.isdir(src):
                shutil.copytree(src, os.path.join(stage, d))
                published.append(d)
        if not published:
            raise SystemExit(f"No {method} vector passed selection in {output_dir}; nothing to publish")
        shutil.copy(report_path, os.path.join(stage, "report.json"))
        if not no_readme:
            with open(os.path.join(stage, "README.md"), "w") as f:
                events_path = os.path.join(output_dir, "events.jsonl")
                examples = pick_examples(events_path) if os.path.exists(events_path) else None
                f.write(discovery_card(report, repo_id, method, examples=examples,
                                       example_prompt=example_prompt))
        create_repo(repo_id, private=private, exist_ok=True)
        upload_folder(folder_path=stage, repo_id=repo_id)
    logger.info(f"Published {', '.join(published)} vectors and report to {repo_id}")


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------

def _load_mlx(model_name: str):
    try:
        from mlx_lm import load
    except ImportError:
        logger.error("The MLX backend needs mlx-lm: pip install 'pts[mlx]'")
        sys.exit(1)
    return load(model_name)


def _load_torch(args):
    from transformers import AutoTokenizer

    from .searchers.base import load_causal_lm, select_device

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return load_causal_lm(args.model, device, dtype=args.dtype), tokenizer


def _load_for_steering(args):
    if args.backend == "mlx":
        return _load_mlx(args.model)
    return _load_torch(args)


def _generate(args, model, tokenizer, prompt: str) -> str:
    if args.backend == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        return generate(
            model, tokenizer, prompt, max_tokens=args.max_tokens,
            sampler=make_sampler(temp=args.temperature),
        )

    import torch

    # A chat-templated prompt already carries its special tokens.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=not args.chat)
    inputs = inputs.to(next(model.parameters()).device)
    sampling = (
        dict(do_sample=True, temperature=args.temperature) if args.temperature > 0
        else dict(do_sample=False)
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def steer(args) -> None:
    setup_logging(args.log_level)

    from .steering import steer as apply_steering
    from .steering.vector import load_vectors

    model, tokenizer = _load_for_steering(args)
    vectors = load_vectors(args.vector)

    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    if args.compare:
        print("=== unsteered ===")
        print(_generate(args, model, tokenizer, prompt))
        print("=== steered ===")
    with apply_steering(model, vectors, model_id=args.model):
        print(_generate(args, model, tokenizer, prompt))


def build_vector(args) -> None:
    setup_logging(args.log_level)

    from .steering import backend_for

    model, tokenizer = _load_for_steering(args)

    def read(path: str) -> List[List[int]]:
        with open(path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        return [tokenizer.encode(line) for line in lines]

    positive, negative = read(args.positive), read(args.negative)
    logger.info(f"{len(positive)} positive and {len(negative)} negative examples")
    vec = backend_for(model).mean_difference(
        model, positive, negative, layers=args.layers,
        base_model=args.model, behavior=args.behavior,
    )
    vec.save(args.output_path)
    for e in vec.entries:
        logger.info(
            f"  layer {e.layer}: |v| = {float((e.vector ** 2).sum()) ** 0.5:.2f}, "
            f"typical residual norm {e.ref_norm:.2f}"
        )
    logger.info(f"Wrote steering vector to {args.output_path}")


def discover(args) -> None:
    setup_logging(args.log_level)

    from .discovery.pipeline import DiscoveryConfig, discover as run
    from .discovery.search import SearchConfig

    model, tokenizer = _load_mlx(args.model)
    config = DiscoveryConfig(
        num_items=args.num_items,
        seed=args.seed,
        layers=args.layers,
        fractions=args.fractions,
        min_answered_ratio=args.min_answered_ratio,
        max_kl=args.max_kl,
        eval_samples=args.eval_samples,
        random_null=args.random_null,
        prompt_increase=args.prompt_increase,
        prompt_decrease=args.prompt_decrease,
        reuse_search=args.reuse_search,
        chat_template_args=json.loads(args.chat_template_args) if args.chat_template_args else {},
        memory_fraction=args.memory_fraction,
        batch_size=args.batch_size,
        sampling={k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                                    ("top_k", args.top_k), ("presence_penalty", args.presence_penalty))
                  if v is not None},
        search=SearchConfig(
            traces_per_item=args.traces_per_item,
            max_tokens=args.max_tokens,
            samples=args.samples,
            min_samples=args.min_samples or args.samples,
            prob_threshold=args.prob_threshold,
            max_estimates_per_trace=args.max_estimates_per_trace,
            min_start=args.min_prob,
            max_start=args.max_prob,
        ),
    )
    behavior = args.behavior or os.path.splitext(os.path.basename(args.eval))[0]
    report = run(model, tokenizer, args.model, args.eval, behavior, args.output_dir, config)
    print(json.dumps({k: report[k] for k in ("behavior", "events", "base")}, indent=2))

    def ci(e):
        return f"{e['delta']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]"

    print("\nTest-set effect on P(behavior), paired, 95% bootstrap CI:")
    for name, m in report["methods"].items():
        for label in ("increase", "decrease"):
            r = m[label]
            if r is None:
                print(f"  {name:26s} {label:8s} no setting passed the damage checks")
                continue
            print(f"  {name:26s} {label:8s} {ci(r['test_effect'])}  "
                  f"random-null p={r['random_null']['p_value']:.2f}  layer {r['layer']} "
                  f"x{r['fraction']:+.2f}  answered {r['test']['answered']:.2f}  KL {r['test']['kl']:.3f}")
    for label, b in report["prompt_baselines"].items():
        print(f"  {'prompt':26s} {label:8s} {ci(b['test_effect'])}  answered {b['test']['answered']:.2f}")
    print("\nPTS advantage over each alternative (positive = PTS moves P further):")
    for key, c in report["comparisons"].items():
        print(f"  {key:56s} {c['advantage']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")


def serve(args) -> None:
    setup_logging(args.log_level)

    from .steering.serving import SteeringRegistry, parse_named, parse_preset

    try:
        registry = SteeringRegistry(
            named=dict(parse_named(v) for v in args.vector or []),
            presets=dict(parse_preset(p) for p in args.preset or []),
            allow_any=args.allow_any_vector,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.backend == "torch":
        _serve_torch(args, registry)
        return
    try:
        from .steering.mlx_server import serve as run
    except ImportError:
        logger.error("pts serve needs mlx-lm: pip install 'pts[mlx]' (or use --backend torch)")
        sys.exit(1)
    run(registry, args.server_args)


def _serve_torch(args, registry) -> None:
    """``pts serve --backend torch``: the same API on transformers."""
    parser = argparse.ArgumentParser(prog="pts serve --backend torch")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    opts = parser.parse_args(args.server_args)

    from .steering.torch_server import TorchSteeringServer, serve as run

    model, tokenizer = _load_torch(opts)
    run(TorchSteeringServer(model, tokenizer, opts.model, registry,
                            max_tokens=opts.max_tokens, temperature=opts.temp), opts.host, opts.port)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _add_latent_args(p) -> None:
    p.add_argument(
        "--readout-method",
        default="logit_lens",
        choices=["jlens", "jlens_cosine", "logit_lens"],
        help="How to read meta-tokens out of the workspace. 'jlens' needs "
             "--jlens-path. 'jlens_cosine' ranks by cosine to each token's J-lens "
             "vector, as WorkspaceBench does; its score is a cosine, not a "
             "probability. 'logit_lens' needs no "
             "calibration but is weaker evidence: it reads what an activation "
             "would say now, not what it pushes the model to say later.",
    )
    p.add_argument(
        "--jlens-path",
        default=None,
        help="A `pts fit-jlens` directory, a reference jacobian-lens .pt file, or "
             "hf://<org>/<repo>/<file.pt> (e.g. a Neuronpedia lens from "
             "hf://neuronpedia/jacobian-lens/...)",
    )
    p.add_argument(
        "--workspace-layers",
        nargs="+",
        default=None,
        help="Layers to read the workspace from, or 'auto' (default)",
    )
    p.add_argument(
        "--workspace-layer-fraction",
        nargs=2,
        type=float,
        default=[0.38, 0.92],
        metavar=("LO", "HI"),
        help="Depth band for auto layer selection (default 0.38 0.92, the band "
             "the workspace paper reports)",
    )
    p.add_argument("--window-before", type=int, default=8,
                   help="How many token positions before an event to read the workspace at")
    # Not --top-k: `pts run` already uses that for sampling.
    p.add_argument("--readout-top-k", type=int, default=25,
                   help="How many vocabulary tokens to read out per position")
    p.add_argument("--min-score", type=float, default=0.01,
                   help="Minimum readout score to keep (a probability, or a cosine "
                        "for jlens_cosine)")
    p.add_argument("--keep-per-position", type=int, default=3,
                   help="Max meta-token events kept per position/layer")
    p.add_argument("--link-threshold", type=float, default=0.5, help="Minimum link score")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pts",
        description="PTS -- pivotal reasoning events across latent, token, and sentence scales",
    )
    sub = parser.add_subparsers(dest="command")

    def add_dtype(p):
        p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"],
                       help="Model precision. 'auto' is bf16 on Ampere+ CUDA, fp16 on "
                            "older CUDA, fp32 on MPS and CPU. Half precision off CUDA "
                            "falls back to fp32 if the device lacks a kernel.")

    def add_common(p):
        p.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    # -- run --
    run_p = sub.add_parser("run", help="Search for pivotal reasoning events")
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--granularity", default="token",
                       choices=["token", "sentence", "latent", "all"],
                       help="Which scale(s) to search")
    run_p.add_argument("--dataset", default="codelion/optillmbench")
    run_p.add_argument("--config", default=None)
    run_p.add_argument("--split", default="train")
    run_p.add_argument("--output-path", default="pts_events.jsonl")
    run_p.add_argument("--device", default=None)
    add_dtype(run_p)
    run_p.add_argument("--task-type", default="generic")
    run_p.add_argument("--prob-threshold", type=float, default=0.2)
    run_p.add_argument("--temperature", type=float, default=0.6)
    run_p.add_argument("--top-p", type=float, default=0.95)
    run_p.add_argument("--top-k", type=int, default=20)
    run_p.add_argument("--min-p", type=float, default=0.0)
    run_p.add_argument("--max-new-tokens", type=int, default=512)
    run_p.add_argument("--num-samples", type=int, default=50)
    run_p.add_argument("--batch-size", type=int, default=5)
    run_p.add_argument("--sample-size", type=int, default=None)
    run_p.add_argument("--max-examples", type=int, default=100)
    run_p.add_argument("--max-generations", type=int, default=10)
    run_p.add_argument("--min-prob", type=float, default=0.2)
    run_p.add_argument("--max-prob", type=float, default=0.8)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--system-prompt", default=None)
    run_p.add_argument("--query-key", default=None)
    run_p.add_argument("--answer-key", default=None)
    run_p.add_argument("--debug", action="store_true")
    run_p.add_argument("--skip-embeddings", action="store_true")
    run_p.add_argument("--enable-verification", action="store_true")
    run_p.add_argument("--generate-thought-anchors", action="store_true",
                       help="Deprecated alias for --granularity sentence")
    _add_latent_args(run_p)
    add_common(run_p)

    # -- migrate --
    mig_p = sub.add_parser("migrate", help="Convert legacy records to the unified event schema")
    mig_p.add_argument("--input-path", required=True)
    mig_p.add_argument("--output-path", required=True)
    add_common(mig_p)

    # -- fit-jlens --
    fit_p = sub.add_parser("fit-jlens", help="Calibrate a Jacobian lens for a model")
    fit_p.add_argument("--model", required=True)
    fit_p.add_argument("--output-path", required=True)
    fit_p.add_argument("--calibration-dataset", default="codelion/optillmbench")
    fit_p.add_argument("--calibration-file", default=None,
                       help="Plain text file, one calibration sequence per line")
    fit_p.add_argument("--split", default="train")
    fit_p.add_argument("--num-sequences", type=int, default=25,
                       help="Calibration sequences (the paper shows ~25 is nearly as "
                            "good as 1000)")
    fit_p.add_argument("--seq-len", type=int, default=128)
    fit_p.add_argument("--basis-chunk", type=int, default=64,
                       help="Output basis directions differentiated per backward call; "
                            "lower this if you hit OOM")
    fit_p.add_argument("--max-layers", type=int, default=4)
    fit_p.add_argument("--device", default=None)
    fit_p.add_argument("--seed", type=int, default=42)
    fit_p.add_argument("--workspace-layers", nargs="+", default=None)
    fit_p.add_argument("--workspace-layer-fraction", nargs=2, type=float,
                       default=[0.38, 0.92], metavar=("LO", "HI"))
    add_common(fit_p)

    # -- enrich --
    enr_p = sub.add_parser("enrich", help="Add latent meta-token events to a dataset")
    enr_p.add_argument("--input-path", required=True)
    enr_p.add_argument("--output-path", required=True)
    enr_p.add_argument("--model", required=True)
    enr_p.add_argument("--device", default=None)
    add_dtype(enr_p)
    enr_p.add_argument("--with-latent", action="store_true",
                       help="Read the workspace around each existing event")
    enr_p.add_argument("--allow-model-mismatch", action="store_true",
                       help="Enrich events produced by a different model (exploratory; "
                            "PTS records are model-specific)")
    enr_p.add_argument("--shuffle-control", action="store_true",
                       help="Report link scores against shuffled queries as a baseline")
    _add_latent_args(enr_p)
    add_common(enr_p)

    # -- link --
    link_p = sub.add_parser("link", help="Link events into causal chains")
    link_p.add_argument("--input-path", required=True)
    link_p.add_argument("--output-path", required=True)
    link_p.add_argument("--edges-output-path", default=None,
                        help="Also write the links as a standalone edge list")
    link_p.add_argument("--window-before", type=int, default=8)
    link_p.add_argument("--link-threshold", type=float, default=0.5)
    link_p.add_argument("--shuffle-control", action="store_true")
    add_common(link_p)

    # -- export --
    exp_p = sub.add_parser("export", help="Render events into a downstream format")
    exp_p.add_argument("--input-path", required=True)
    exp_p.add_argument("--output-path", required=True)
    exp_p.add_argument(
        "--format",
        required=True,
        choices=["causal_events", "metatokens", "pivotal_tokens", "thought_anchors",
                 "dpo", "steering"],
    )
    exp_p.add_argument("--model", default=None)
    exp_p.add_argument("--dataset", default=None,
                       help="Source dataset, needed to rebuild a real oracle for "
                            "--find-rejected-tokens")
    exp_p.add_argument("--split", default="train")
    exp_p.add_argument("--device", default=None)
    add_dtype(exp_p)
    exp_p.add_argument("--min-prob-delta", type=float, default=0.1)
    exp_p.add_argument("--min-score", type=float, default=0.0)
    exp_p.add_argument("--max-pairs", type=int, default=None)
    exp_p.add_argument("--balance", action="store_true")
    exp_p.add_argument("--seed", type=int, default=42)
    exp_p.add_argument("--num-candidates", type=int, default=10)
    exp_p.add_argument("--find-rejected-tokens", action="store_true")
    exp_p.add_argument("--layer-nums", type=int, nargs="+", default=[19, 23, 27])
    exp_p.add_argument("--select-layer", type=int, default=None)
    exp_p.add_argument("--num-clusters", type=int, default=10)
    exp_p.add_argument("--pca-components", type=int, default=50)
    exp_p.add_argument("--batch-size", type=int, default=4)
    exp_p.add_argument("--hf-push", action="store_true")
    exp_p.add_argument("--hf-repo-id", default=None)
    exp_p.add_argument("--private", action="store_true")
    add_common(exp_p)

    # -- push --
    push_p = sub.add_parser("push", help="Upload a dataset, or a pts discover output directory, to Hugging Face")
    push_p.add_argument("--input-path", required=True)
    push_p.add_argument("--hf-repo-id", required=True)
    push_p.add_argument("--private", action="store_true")
    push_p.add_argument("--no-readme", action="store_true")
    push_p.add_argument("--example-prompt", default=None,
                        help="Discover runs: the prompt the model card's usage examples use")
    push_p.add_argument("--model", default=None)
    add_common(push_p)

    # -- steer --
    steer_p = sub.add_parser("steer", help="Generate with steering vectors applied")
    steer_p.add_argument("--model", required=True)
    steer_p.add_argument("--vector", action="append", required=True,
                         help="A steering vector directory, .safetensors file, or "
                              "hf://<org>/<repo>[/<path>], optionally with ':<alpha>'. "
                              "Repeat to stack vectors; their additions sum.")
    steer_p.add_argument("--prompt", required=True)
    steer_p.add_argument("--chat", action="store_true",
                         help="Wrap the prompt in the model's chat template")
    steer_p.add_argument("--max-tokens", type=int, default=128)
    steer_p.add_argument("--temperature", type=float, default=0.0)
    steer_p.add_argument("--compare", action="store_true",
                         help="Also print the unsteered generation")
    steer_p.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    steer_p.add_argument("--device", default=None, help="torch backend only")
    add_dtype(steer_p)
    add_common(steer_p)

    # -- build-vector --
    bv_p = sub.add_parser("build-vector",
                          help="Build a mean-difference steering vector from paired examples")
    bv_p.add_argument("--model", required=True)
    bv_p.add_argument("--positive", required=True,
                      help="Text file, one example per line, showing the behavior")
    bv_p.add_argument("--negative", required=True,
                      help="Text file, one example per line, not showing it")
    bv_p.add_argument("--layers", type=int, nargs="+", required=True)
    bv_p.add_argument("--behavior", default=None)
    bv_p.add_argument("--output-path", required=True,
                      help="A directory, or a .safetensors file for a single-file vector")
    bv_p.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    bv_p.add_argument("--device", default=None, help="torch backend only")
    add_dtype(bv_p)
    add_common(bv_p)

    # -- discover --
    disc_p = sub.add_parser(
        "discover",
        help="Find pivotal tokens for a behavior and turn them into validated steering vectors",
    )
    disc_p.add_argument("--model", required=True)
    disc_p.add_argument("--eval", required=True,
                        help="Multiple-choice behavior eval (JSONL, or hf://<org>/<dataset>/<file>), "
                             "e.g. hf://Anthropic/model-written-evals/sycophancy/"
                             "sycophancy_on_political_typology_quiz.jsonl")
    disc_p.add_argument("--behavior", default=None, help="Name for the behavior (default: file name)")
    disc_p.add_argument("--output-dir", required=True)
    disc_p.add_argument("--num-items", type=int, default=120)
    disc_p.add_argument("--layers", type=int, nargs="+", default=None)
    disc_p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4],
                        help="Steering strengths to sweep, as fractions of the residual norm")
    disc_p.add_argument("--min-answered-ratio", type=float, default=0.9,
                        help="Reject steering that makes the model give a recognisable answer less "
                             "than this fraction as often as unsteered")
    disc_p.add_argument("--max-kl", type=float, default=0.5)
    disc_p.add_argument("--traces-per-item", type=int, default=10)
    disc_p.add_argument("--max-tokens", type=int, default=512)
    disc_p.add_argument("--samples", type=int, default=50,
                        help="Most sampled completions per probability estimate in the search")
    disc_p.add_argument("--min-samples", type=int, default=10,
                        help="Completions an estimate starts with; more are drawn only while a "
                             "bisection comparison is undecided (0 or = --samples: fixed size)")
    disc_p.add_argument("--prob-threshold", type=float, default=0.2)
    disc_p.add_argument("--max-estimates-per-trace", type=int, default=0,
                        help="Bisection budget per trace (0: no limit, as in token PTS)")
    disc_p.add_argument("--min-prob", type=float, default=0.2,
                        help="Search only questions whose sampled P(behavior) is in "
                             "[--min-prob, --max-prob]")
    disc_p.add_argument("--max-prob", type=float, default=0.8)
    disc_p.add_argument("--chat-template-args", default=None,
                        help='JSON passed to the chat template, as in mlx_lm, e.g. '
                             '\'{"enable_thinking": false}\'')
    disc_p.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default 0.7); use the model card's settings")
    disc_p.add_argument("--top-p", type=float, default=None)
    disc_p.add_argument("--top-k", type=int, default=None)
    disc_p.add_argument("--presence-penalty", type=float, default=None,
                        help="e.g. 1.5 for Qwen3.5 thinking mode, which otherwise can loop")
    disc_p.add_argument("--memory-fraction", type=float, default=0.5,
                        help="Cap MLX memory at this fraction of the device's (lower it when "
                             "sharing the machine with another model)")
    disc_p.add_argument("--batch-size", type=int, default=32,
                        help="Sequences generated at once. Only speed and memory change: a batch's KV "
                             "cache is padded to its longest sequence (default: 32)")
    disc_p.add_argument("--eval-samples", type=int, default=16,
                        help="Sampled completions per test question when measuring steering")
    disc_p.add_argument("--random-null", type=int, default=5,
                        help="Random vectors of the same size tested at each chosen setting")
    disc_p.add_argument("--prompt-increase", default=None,
                        help="Instruction for the prompting baseline (default: 'Answer in a way that "
                             "shows <behavior>.')")
    disc_p.add_argument("--prompt-decrease", default=None)
    disc_p.add_argument("--reuse-search", action="store_true",
                        help="Use events.jsonl and traces.jsonl in --output-dir as they are, even if "
                             "the search settings differ (a rerun with the same settings resumes "
                             "automatically)")
    disc_p.add_argument("--seed", type=int, default=0)
    add_common(disc_p)

    # -- serve --
    serve_p = sub.add_parser(
        "serve",
        help="OpenAI-compatible server with steering (MLX or PyTorch)",
        description="With --backend mlx (default) every flag not listed here is passed to "
                    "mlx_lm.server (--model, --host, --port, --max-tokens, ...). With "
                    "--backend torch: --model, --host, --port, --device, --dtype, "
                    "--max-tokens, --temp.",
    )
    serve_p.add_argument("--vector", action="append", metavar="NAME=PATH",
                         help="Register a steering vector requests can name. Repeatable.")
    serve_p.add_argument("--preset", action="append", metavar="NAME=VEC[:ALPHA][,...][@MODEL]",
                         help="A steering configuration clients select as the model name. "
                              "Repeatable; listed by /v1/models.")
    serve_p.add_argument("--backend", default="mlx", choices=["mlx", "torch"],
                         help="mlx: mlx_lm.server with batching (Apple silicon). torch: "
                              "transformers, one request at a time (CUDA, MPS, CPU).")
    serve_p.add_argument("--allow-any-vector", action="store_true",
                         help="Let requests pass vector paths and hf:// ids, not just "
                              "registered names. The server will read and download them.")
    add_common(serve_p)

    args, extra = parser.parse_known_args(argv)
    if args.command == "serve":
        args.server_args = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args, parser


def main(argv=None) -> None:
    args, parser = parse_args(argv)

    handlers = {
        "run": run_pts,
        "migrate": migrate,
        "fit-jlens": fit_jlens,
        "enrich": enrich,
        "link": link,
        "export": export,
        "push": push_to_hf,
        "steer": steer,
        "build-vector": build_vector,
        "serve": serve,
        "discover": discover,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()

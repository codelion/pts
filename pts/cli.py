"""
Command-line interface for PTS.

    pts run       search for pivotal events at one or more scales
    pts enrich    add latent meta-token events to an existing dataset
    pts fit-jlens calibrate a Jacobian lens for a model
    pts link      link latent, token, and sentence events into causal chains
    pts migrate   convert legacy pivotal-token / thought-anchor files to PTS events
    pts export    render events into a downstream format
    pts push      upload a dataset to Hugging Face
"""

import argparse
import json
import logging
import os
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .latent.activations import get_layer_modules, resolve_workspace_layers
    from .latent.jlens import JLens
    from .searchers.base import select_device

    device = select_device(args.device)
    logger.info(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float32
    # Fitting differentiates through the network; fp16 Jacobians underflow badly,
    # so calibration runs in fp32 regardless of what inference would use.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .event_storage import EventStorage
    from .latent.jlens import load_readout
    from .latent.metatokens import enrich_events_with_latent
    from .linking import EventLinker
    from .searchers.base import select_device

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device in ("cpu", "mps") else torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

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
    )


def _push(
    path: str,
    repo_id: str,
    private: bool,
    model_name: Optional[str],
    file_type: Optional[str] = None,
    no_readme: bool = False,
) -> None:
    from huggingface_hub import create_repo, upload_file

    from .exporters import detect_file_type, generate_dataset_card

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


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _add_latent_args(p) -> None:
    p.add_argument(
        "--readout-method",
        default="logit_lens",
        choices=["jlens", "logit_lens"],
        help="How to read meta-tokens out of the workspace. 'jlens' needs "
             "--jlens-path (fit one with `pts fit-jlens`). 'logit_lens' needs no "
             "calibration but is weaker evidence: it reads what an activation "
             "would say now, not what it pushes the model to say later.",
    )
    p.add_argument("--jlens-path", default=None, help="Directory holding fitted J-lens matrices")
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
    p.add_argument("--min-score", type=float, default=0.01, help="Minimum readout score to keep")
    p.add_argument("--keep-per-position", type=int, default=3,
                   help="Max meta-token events kept per position/layer")
    p.add_argument("--link-threshold", type=float, default=0.5, help="Minimum link score")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pts",
        description="PTS -- pivotal reasoning events across latent, token, and sentence scales",
    )
    sub = parser.add_subparsers(dest="command")

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
    push_p = sub.add_parser("push", help="Upload a dataset to Hugging Face")
    push_p.add_argument("--input-path", required=True)
    push_p.add_argument("--hf-repo-id", required=True)
    push_p.add_argument("--private", action="store_true")
    push_p.add_argument("--no-readme", action="store_true")
    push_p.add_argument("--model", default=None)
    add_common(push_p)

    return parser.parse_args(argv), parser


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
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()

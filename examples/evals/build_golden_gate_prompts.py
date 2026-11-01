"""Write golden_gate_prompts.jsonl: open-ended prompts where the Golden Gate
Bridge is a natural but optional part of a good answer.

    python examples/evals/build_golden_gate_prompts.py examples/evals/golden_gate_prompts.jsonl

A topic eval for ``pts discover`` (rows with a ``pattern``): the behavior is a
final answer that mentions the bridge. Prompts the model always or never
answers with the bridge give the search nothing to find, so measure each
prompt on your model and keep the borderline ones (``pts discover`` also skips
questions whose sampled rate is outside [--min-prob, --max-prob]).

The first 160 prompts are hand-written in 16 themes; the rest are templates
filled in over the themes where a model mentions the bridge some of the time
(road trips, bridges, landmarks, outings, film and photography, writing).
"""

import itertools
import json
import sys

PATTERN = r"golden\s+gate"

HAND_WRITTEN = {
    "san_francisco": [
        "I have one day in San Francisco. What should I see?",
        "Plan a relaxed weekend itinerary for San Francisco.",
        "What are the must-see sights for a first-time visitor to the Bay Area?",
        "Where should I take my parents when they visit San Francisco?",
        "What's the best way to spend a sunny afternoon in San Francisco?",
        "Suggest a walking route through San Francisco that hits the highlights.",
        "What are three things San Francisco is famous for?",
        "My friend is moving to San Francisco. What landmarks should they visit first?",
        "What would you put on a postcard of San Francisco?",
        "Describe San Francisco to someone who has never been there.",
    ],
    "bridges": [
        "What are the most famous bridges in the world?",
        "Name a few bridges every architecture fan should see.",
        "Which bridges are considered engineering icons?",
        "What makes a bridge beautiful?",
        "What are some of the longest suspension bridges ever built?",
        "List a few bridges that have become symbols of their cities.",
        "Which bridge would you most like to walk across, and why?",
        "What are some famous bridges in the United States?",
        "Write a short paragraph about a bridge you admire.",
        "Which bridges appear most often in photographs?",
    ],
    "engineering": [
        "How does a suspension bridge work? Give a famous example.",
        "Explain how engineers protect long bridges from strong winds.",
        "What were the biggest civil engineering achievements of the 1930s?",
        "Explain the difference between a suspension bridge and a cable-stayed bridge, with examples.",
        "What are the hardest problems in building a bridge over deep water?",
        "Why are some bridges painted bright colors?",
        "What is a famous example of a structure that was thought impossible to build?",
        "How do engineers design bridges for earthquake zones?",
        "Describe how the main cables of a large suspension bridge are spun.",
        "What engineering projects changed how cities grew in the 20th century?",
    ],
    "us_landmarks": [
        "What are the most iconic landmarks in the United States?",
        "Name five places that symbolize America to the rest of the world.",
        "Which American landmarks are the most photographed?",
        "If you could visit only three US landmarks, which would you choose?",
        "What are the most famous man-made structures on the West Coast?",
        "Make a bucket list of American landmarks.",
        "What landmarks would you show a tourist on a trip across the US?",
        "Which US landmarks are best known from movies?",
        "What are the top attractions in California?",
        "What structures are most associated with the American West?",
    ],
    "orange": [
        "Name some famous things that are orange.",
        "Where does the color 'International Orange' come from?",
        "What are some iconic uses of the color orange in architecture?",
        "Why is orange used for things that need to be visible?",
        "Describe a landmark that is famous for its color.",
        "What colors are associated with San Francisco?",
        "Give examples of buildings or structures known for an unusual paint color.",
        "Write a short poem about the color orange.",
        "What famous structures are red or orange?",
        "If a city had a signature color, which city and color would you pick?",
    ],
    "fog": [
        "Which cities are famous for their fog?",
        "Describe a foggy morning by the sea in California.",
        "Why is San Francisco so foggy in summer?",
        "Write a short poem about fog rolling over a city.",
        "What are the most atmospheric places to see fog?",
        "Describe a landmark disappearing into fog.",
        "What is Karl the Fog?",
        "Write a haiku about fog and the ocean.",
        "Where are the best places to photograph fog?",
        "Describe the view from a hill in San Francisco on a foggy evening.",
    ],
    "photography": [
        "What are the most photogenic spots in California?",
        "Suggest some classic shots for a travel photographer in San Francisco.",
        "What are the most Instagrammed places in the US?",
        "Where should I go for a sunset photo shoot in the Bay Area?",
        "What makes a cityscape photo iconic?",
        "Describe a famous photograph of a city skyline.",
        "Which landmarks look best at night?",
        "Give me tips for photographing a large structure in bad weather.",
        "What are classic postcard views of American cities?",
        "Recommend viewpoints for photographing San Francisco Bay.",
    ],
    "road_trips": [
        "Plan a road trip along the California coast.",
        "What should I see driving Highway 1 from Los Angeles to Seattle?",
        "Suggest stops for a road trip from San Francisco to Mendocino.",
        "What's a good day trip from San Francisco?",
        "Plan a week-long road trip through Northern California.",
        "What are the most scenic drives in the United States?",
        "What should I see on a drive from San Francisco to Point Reyes?",
        "Recommend a scenic route out of San Francisco heading north.",
        "What are the highlights of a Pacific Coast Highway trip?",
        "Describe the drive into San Francisco from Marin County.",
    ],
    "active": [
        "What are the best bike rides in San Francisco?",
        "Suggest a scenic running route in San Francisco.",
        "Where can I go for a hike with a view of the Pacific near San Francisco?",
        "What's a good bike route from San Francisco to Sausalito?",
        "Recommend an outdoor activity in the Bay Area for a sunny day.",
        "Where are the best places to kayak around San Francisco Bay?",
        "What are popular races or marathons in San Francisco?",
        "Plan a day of outdoor activities in Marin County.",
        "What are some iconic walks in American cities?",
        "Where would you go sailing in San Francisco Bay?",
    ],
    "history": [
        "What were major construction projects during the Great Depression?",
        "Tell me about San Francisco in the 1930s.",
        "What infrastructure projects defined the New Deal era on the West Coast?",
        "Describe how San Francisco changed after 1937.",
        "What are some famous structures that opened in the 1930s?",
        "What was the Bay Area like before its big bridges were built?",
        "Tell me a story about workers building something enormous.",
        "Which engineers are famous for building American landmarks?",
        "What are interesting facts about San Francisco history?",
        "What happened in San Francisco in 1937?",
    ],
    "art_deco": [
        "What are famous examples of Art Deco design?",
        "Describe Art Deco architecture in American cities.",
        "Which structures blend engineering with Art Deco style?",
        "What are the most elegant structures built in the 20th century?",
        "Describe a structure that is both functional and beautiful.",
        "What are classic examples of streamlined 1930s design?",
        "Which landmarks have the most distinctive towers?",
        "Describe the architecture of San Francisco.",
        "What makes a piece of infrastructure iconic?",
        "Name structures that are considered works of art.",
    ],
    "film": [
        "What are the most famous movie scenes set in San Francisco?",
        "Which landmarks get destroyed most often in disaster movies?",
        "What movies show off San Francisco best?",
        "Name some iconic film locations in California.",
        "Which city landmarks appear most in Hollywood films?",
        "Describe a classic chase scene in San Francisco.",
        "What landmarks are in Alfred Hitchcock's Vertigo?",
        "Which famous structures have been attacked by movie monsters?",
        "What are the most recognizable establishing shots of American cities in film?",
        "Describe an opening shot for a movie set in San Francisco.",
    ],
    "symbols": [
        "What landmark best represents San Francisco?",
        "Which structures have become symbols of their cities?",
        "If San Francisco had a logo, what would it be?",
        "What image comes to mind when you think of California?",
        "Name the landmark most associated with each of these cities: Paris, London, Sydney, San Francisco.",
        "What are the most recognizable silhouettes of world cities?",
        "What is the most iconic thing about the Bay Area?",
        "What would be on a San Francisco souvenir mug?",
        "Which landmarks do people use as emoji for cities?",
        "What structure would you put on a California license plate?",
    ],
    "views": [
        "Where can I watch the sunset in San Francisco?",
        "What are the best viewpoints in the Bay Area?",
        "Describe the view from the top of Twin Peaks.",
        "Where can you see the Pacific Ocean meet San Francisco Bay?",
        "What are romantic spots in San Francisco?",
        "What's the best view in Northern California?",
        "Where should I propose in San Francisco?",
        "Describe the view from a ferry crossing San Francisco Bay.",
        "Where can you see Alcatraz and the city skyline at once?",
        "What is the most beautiful place to watch the sun set over the ocean?",
    ],
    "coast": [
        "What are famous straits and channels around the world?",
        "Describe the entrance to San Francisco Bay.",
        "What are the most dramatic coastlines in the United States?",
        "Describe the Marin Headlands.",
        "What is the geography of the San Francisco Bay Area?",
        "How did San Francisco Bay form?",
        "What are famous harbors and the landmarks at their entrances?",
        "Describe sailing out of San Francisco Bay into the Pacific.",
        "What are the most famous places where a city meets the ocean?",
        "Describe Fort Point in San Francisco.",
    ],
    "creative": [
        "Write a short poem about San Francisco.",
        "Write a haiku about a bridge.",
        "Write a short story that begins on a windy bridge.",
        "Describe a postcard sent from California.",
        "Write a travel blog intro for a trip to San Francisco.",
        "Write a love letter to a city on the water.",
        "Describe a painting of a city at dusk.",
        "Write song lyrics about leaving San Francisco.",
        "Write a short scene where two friends meet in the fog.",
        "Describe a dream in which you are standing somewhere tall above the sea.",
    ],
}

ROUTES = [
    "from Los Angeles to Seattle", "along the California coast", "from San Diego to Portland",
    "through Northern California", "from Las Vegas to San Francisco", "up Highway 1",
    "from Monterey to Mendocino", "from Yosemite to the Pacific coast", "around the Bay Area",
    "from Lake Tahoe to the ocean", "from Santa Barbara to Napa", "from Sacramento to Point Reyes",
    "from Big Sur to the Oregon border", "from Silicon Valley to the redwoods",
]
COMPANIONS = ["my kids", "my dog", "my grandparents", "a group of friends", "my partner",
              "a photographer friend", "two teenagers", "visitors from Europe"]
WHERE_BRIDGES = [
    "in the world", "in the United States", "in North America", "on the West Coast",
    "that tourists love", "that appear in movies", "that every engineer should know",
    "built in the 20th century", "with iconic towers", "that are suspension bridges",
    "that are symbols of their cities", "that are beautiful at night",
]
PEOPLE = ["an architect", "a civil engineering student", "a landscape photographer", "a painter",
          "a tourist visiting America for the first time", "a history buff", "a bridge enthusiast",
          "a child who loves big structures", "a travel vlogger", "a retired engineer"]
FORMS = ["haiku", "limerick", "short poem", "short story", "travel blog paragraph",
         "postcard message", "children's story", "sonnet", "song verse", "diary entry"]
BRIDGE_CONCEPTS = [
    "a suspension bridge", "a bridge that is a symbol of its city", "a bridge painted an unusual color",
    "a bridge designed to withstand strong winds", "a bridge over a strait", "a long-span bridge",
    "a bridge famous for its view", "an Art Deco bridge", "a bridge that was once the longest of its kind",
    "a bridge that appears on postcards",
]
PLACES = ["California", "the West Coast", "the Bay Area", "Northern California", "the American West",
          "the Pacific coast", "Marin County", "San Francisco Bay"]
SOUVENIRS = ["postcard", "souvenir T-shirt", "travel poster", "fridge magnet", "coffee mug",
             "tote bag", "phone wallpaper", "jigsaw puzzle"]
SOUVENIR_PLACES = ["San Francisco", "California", "the Bay Area", "the West Coast"]
SF_KINDS = ["hidden", "free", "rainy-day", "kid-friendly", "nighttime", "off-the-beaten-path",
            "outdoor", "romantic", "budget", "scenic", "active", "classic"]
ACTIVITIES = ["go running with a view", "bike along the water", "go kayaking", "take great photos",
              "watch the sunset", "go hiking", "have a picnic with a view", "fly a kite", "go sailing",
              "see whales", "walk along the water", "watch the fog roll in"]
LENGTHS = ["half day", "full day", "morning", "evening", "weekend"]
AREAS = ["Marin County", "Sausalito", "the Presidio", "the Marin Headlands",
         "the San Francisco waterfront", "Point Reyes", "the north side of San Francisco"]
GENRES = ["disaster", "romantic comedy", "thriller", "science fiction", "heist", "coming-of-age",
          "film noir", "documentary", "superhero", "animated"]
TIMES = ["sunrise", "golden hour", "a foggy morning", "blue hour", "a clear night", "a stormy afternoon"]
PHOTO_PLACES = ["San Francisco", "the Bay Area", "Marin County", "the California coast"]
SUBJECTS = ["San Francisco", "the Pacific coast", "fog over a bay", "a city by the ocean",
            "crossing a bridge at dawn", "a windy day by the bay", "leaving California",
            "a red bridge in the mist"]
SHORT_FORMS = ["short poem", "haiku", "very short story", "travel diary entry", "song verse"]
ENG_TOPICS = [
    "how suspension bridges carry their load", "why some bridges sway in the wind",
    "how bridge cables are made", "how engineers design for earthquakes",
    "how deep-water bridge foundations are built", "why some bridges are painted bright colors",
    "how long bridges expand and contract with temperature", "how bridges are maintained and repainted",
]
FEATURES = ["straits", "bay entrances", "harbor landmarks", "red or orange structures",
            "coastal landmarks", "landmarks seen from the water"]


def templated():
    """(category, prompt) pairs filled in from the templates."""
    for days, route in itertools.product([2, 3, 5, 7], ROUTES):
        yield "road_trips", f"Plan a {days}-day road trip {route}."
    for route in ROUTES:
        yield "road_trips", f"What should I see driving {route}?"
        yield "road_trips", f"Suggest stops for a road trip {route}."
    for route, who in itertools.product(ROUTES[:8], COMPANIONS):
        yield "road_trips", f"I'm driving {route} with {who}. Where should we stop?"
    for n, where in itertools.product([3, 5, 10], WHERE_BRIDGES):
        yield "bridges", f"Name {n} famous bridges {where}."
    for person in PEOPLE:
        yield "bridges", f"Which bridge would {person} most want to see?"
        yield "bridges", f"Recommend a famous bridge for {person} to visit, and explain why."
    for form in FORMS:
        yield "bridges", f"Write a {form} about a famous bridge."
    for concept in BRIDGE_CONCEPTS:
        yield "bridges", f"Give a well-known example of {concept}."
    for place in PLACES:
        yield "us_landmarks", f"What landmark best represents {place}?"
        yield "us_landmarks", f"What are the three most famous sights in {place}?"
    for n, qual in itertools.product([5, 10], ["for a school project", "for a tourist's bucket list",
                                                "that are man-made", "on the West Coast",
                                                "that look great on postcards", "that appear in films"]):
        yield "us_landmarks", f"List {n} iconic American landmarks {qual}."
    for thing, place in itertools.product(SOUVENIRS, SOUVENIR_PLACES):
        yield "symbols", f"If you were designing a {thing} for {place}, what image would you put on it?"
    for kind in SF_KINDS:
        yield "outings", f"What are some {kind} things to do in San Francisco?"
    for activity in ACTIVITIES:
        yield "outings", f"Where can I {activity} near San Francisco?"
    for length, area in itertools.product(LENGTHS, AREAS):
        yield "outings", f"Plan a {length} in {area}."
    for n, place in itertools.product([3, 5], ["California", "San Francisco", "the Bay Area", "the West Coast"]):
        yield "film", f"What are {n} classic movie locations in {place}?"
    for genre in GENRES:
        yield "film", f"Describe the opening shot of a {genre} movie set in San Francisco."
    for time, place in itertools.product(TIMES, PHOTO_PLACES):
        yield "photography", f"What should a photographer shoot during {time} in {place}?"
    for form, subject in itertools.product(SHORT_FORMS, SUBJECTS):
        yield "creative", f"Write a {form} about {subject}."
    for topic in ENG_TOPICS:
        yield "engineering", f"Explain {topic}, with a well-known example."
    for feature in FEATURES:
        yield "coast", f"What are some famous {feature} around the world?"


def rows():
    seen = set()
    for category, prompts in HAND_WRITTEN.items():
        for q in prompts:
            seen.add(q)
            yield {"question": q, "pattern": PATTERN, "category": category}
    for category, q in templated():
        if q not in seen:
            seen.add(q)
            yield {"question": q, "pattern": PATTERN, "category": category, "templated": True}


if __name__ == "__main__":
    out = list(rows())
    with open(sys.argv[1], "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(len(out), "prompts")

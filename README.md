# FullCourt: an NBA research assistant

FullCourt answers questions like "who are the efficient scoring guards that can also run an offense?"
by combining two ways of searching the same season of data: exact statistics in a SQL database, and
a written style profile for every player that can be searched by meaning. Every number you see
comes from the database, and every claim about playing style links back to the profile it came
from.

I built it as a portfolio project to explore hybrid retrieval, meaning SQL and vector search working
together, and to see how far you can keep an LLM honest by only letting it talk about evidence it
was handed. It covers one season (2025-26) and doesn't train any models.

## How a question flows through the app

1. **Parsing.** The question is turned into a structured search plan: season, positions, explicit
   numeric filters, ranking preferences and a short description of the playing style to look for.
   An LLM does this when one is available. Otherwise a rule-based parser takes over.
2. **Stats lookup.** SQL applies the filters and pulls each candidate's numbers.
3. **Style search.** The style description is embedded and compared against each player's
   stat-based style profile, stored in a FAISS index.
4. **Ranking.** Each player gets a statistical fit score and a style relevance score, blended 60/40
   into a final score. You can see the full breakdown for every result.
5. **Explanation.** The LLM writes a short explanation from an evidence package for each player. A
   grounding check rejects anything that mentions a number or source it wasn't given, and a plain
   template is used instead.
6. **UI.** A Streamlit page shows the ranking, key stats, the score breakdown and the source
   passages with links.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/import_stats.py          # downloads 2025-26 stats from Basketball-Reference into SQLite
python scripts/build_stat_profiles.py   # writes a stat-based style profile for every player
python scripts/build_index.py           # embeds all passages and builds the FAISS index

streamlit run app.py
pytest
```

I've tested this with Python 3.14 on an M2 Mac. The first run downloads the embedding model (about
130 MB) from Hugging Face.

### Choosing an LLM

The LLM is optional. Copy `.env.example` to `.env` and set up one or both of these:

- **Gemini.** Add your `GEMINI_API_KEY`. Whenever a key is present, Gemini is used first. The model
  defaults to `gemini-flash-latest`, and you can change it with `FULLCOURT_GEMINI_MODEL`. If Gemini
  answers "overloaded" (503) or "rate limited" (429), the app waits and retries up to three times
  (after about 1, 2 and 4 seconds) before moving on to the backup.
- **Ollama.** Pull a model with [Ollama](https://ollama.com) and set `FULLCOURT_LLM_MODEL` to its
  name. With a Gemini key also set, Ollama acts as the backup: if a Gemini call fails, for example
  because you hit a quota, the same request goes to Ollama. The app requests an 8192-token context
  window from Ollama, since its default is too small for the explanation prompt.

Both providers are asked for output that matches a JSON schema, and the result is validated before
anything uses it.

If you run Ollama, choose a model that fits comfortably in memory alongside the app. On an 8 GB
machine, a 7B or 8B model will swap heavily and take minutes per question; a 3B or 4B model is a
better fit.

If nothing is available, nothing breaks. The app switches to the rule-based parser and template
explanations and tells you so in the sidebar. Set `FULLCOURT_LLM_PROVIDER=none` to turn every LLM
off, including Gemini. There's also an Anthropic adapter (`FULLCOURT_LLM_PROVIDER=anthropic`, plus
`pip install anthropic` and an API key).

With Gemini Flash, parsing a question takes about 5 seconds, and writing explanations for ten
results takes 20 seconds or more. The very first call after startup is slower.

## Where the data comes from

**Stats** come from four Basketball-Reference tables for the 2025-26 regular season, covering all
582 players:

- [per-game](https://www.basketball-reference.com/leagues/NBA_2026_per_game.html) and advanced
  stats for the basics (points, assists, true shooting, usage and so on)
- [shooting](https://www.basketball-reference.com/leagues/NBA_2026_shooting.html), which says where a
  player's shots come from (at the rim, short range, mid-range, threes, corner threes, dunks) and
  how many of their makes were assisted
- [play-by-play](https://www.basketball-reference.com/leagues/NBA_2026_play-by-play.html), which
  estimates how a player's minutes split across the five positions and counts shooting fouls drawn,
  and-ones, charges drawn and the kinds of turnovers they commit

When a player was traded, I keep their combined season row and record each team they played for.

**Style profiles** are short write-ups built for every player from their own numbers by a fixed
template. A profile reads like "83% of shots at the rim, 33% dunks; 77% of made twos assisted",
then says how the player compares with rotation players ("very high: dunk rate, assisted twos;
very low: usage, three-point volume") and ends with style tags such as "rim-running finisher and
lob threat". The comparisons and tags come from simple percentile rules (in
`src/stats/profiles.py`), and players with too small a sample get neither. Profiles leave out the
player's name, so a "similar to Nikola Jokić" search matches playing style rather than other
players named Nikola.

No style text is written by an LLM. The profiles restate numbers, not scouting, and the UI and
explanations label them that way.

A few practical notes:

- Percentages and rates are stored as fractions, so a 61.6% true shooting mark is `0.616`.
  Basketball-Reference reports usage, assist, turnover, rebound, steal and block rates and the
  position estimates in percentage points, so the import divides those by 100.
- The play-by-play counts (fouls drawn, charges, turnover types) are season totals at the source,
  so the import converts them to per-game numbers.
- Missing values stay `NULL`. They're never filled in with zero.
- The scraped pages, the database and the index aren't committed. The scripts rebuild them.
  Basketball-Reference data is used here for a non-commercial demo, so check their terms before
  redistributing it.

### Why the style text comes from stats

The first version used the "Player profile" sections of Wikipedia articles. Only 48 of the 406
rotation players had one, almost all of them stars, so role players had no style evidence and sank
in the rankings. Building a profile from each player's own numbers covers all 582 players. Now
"rim-running bigs who catch lobs" returns centers like Drew Eubanks, Neemias Queta and Jarrett
Allen, and "3-and-D wings" turns up role players like Reed Sheppard, Royce O'Neale and Matisse
Thybulle.

The cost is independence. Wikipedia text was evidence the stats didn't already contain; the
profiles are the stats in words. What the style search adds now is matching by meaning: a question
about "lob threats" finds players whose profiles say "dunks" and "assisted finishes" without anyone
mapping those words to columns by hand.

## How the pieces work

### Parsing

The search plan separates two things that are easy to blur together:

- **Hard filters** come only from numbers the user actually said. "At least 20 points" becomes
  `points_per_game >= 20`.
- **Preferences** are everything vaguer. "Efficient" means rank by true shooting; it never turns
  into an invented cutoff like 60%. Preferences can point up ("good passer"), down ("without
  scoring a lot") or toward the middle ("average playmaker").

The LLM can only pick metrics from a fixed list, and each one maps to a known database column. It
never writes SQL, and filter values are always passed as query parameters. If the question leaves
out the season, the plan uses 2025-26 and marks it as a default. Requests the data can't answer,
like clutch stats or salaries, are listed back to the user instead of being quietly ignored.

### Stats and eligibility

On top of the user's filters, there's a default eligibility floor of 20 games and 10 minutes per
game, so a player who shot 3 for 3 in two games doesn't top an efficiency ranking. The UI shows this
floor as a default you can switch off, separate from anything the user asked for.

### Style search

What gets embedded is not the full profile but a short summary of it with no numbers: the player's
role ("center, big man"), the comparisons with rotation players, and the style tags. My first
version embedded the full profile, and every player came out looking almost the same. The
embedding model barely reads numbers, so the shared template wording dominated: two different
players' profiles scored 0.98 to 0.997 against each other, and a question's scores across all 582
players varied by only about 0.04. Style search was ranking on noise. The number-free summaries
spread that out about four times wider, and "rim-running bigs" went from returning stretch forwards
to returning centers like Kel'el Ware, Branden Carlson and Drew Eubanks.

The summaries are embedded with `BAAI/bge-small-en-v1.5` and stored in a FAISS inner-product
index, so scores are cosine similarities. For "players similar to X" questions, the search uses X's
own summary vector instead of the question text.

The UI doesn't show raw cosine similarity, because it isn't meaningful on its own: similar players
score around 0.99 against each other, and questions land between about 0.6 and 0.75. Instead each
profile shows its style match, the same 0 to 1 score the ranking uses.

Style scores are scaled from 0 to 1 across the candidates. The code scales each type of text
separately and averages them, which only matters if another source of text is ever added: prose
and number-heavy profiles land in different similarity ranges, and comparing them directly would
reward players just for having the extra text.

### Ranking

Each preference maps to a fixed set of stats. Playmaking, for example, is half assist rate and half
assists per game. When two stats measure the same thing, they're averaged rather than counted
twice.

Stats are converted to percentiles within the filtered pool of candidates, so different scales
become comparable and outliers don't dominate. Turnover rate is flipped so that lower counts as
better. The statistical fit is the average of the preference scores. Style scores are scaled from 0
to 1 across the pool, and the final score is 0.6 times the statistical fit plus 0.4 times the style
score. Those weights are a starting point, not something I've tuned.

Some goals care about which direction a stat points. A low share of assisted baskets means a player
creates their own shot, while a high share on threes means a catch-and-shoot role, so each goal
says which of its stats count in reverse. The UI shows the plain percentile and notes "lower is
better here" when that applies.

The goals the shooting and play-by-play data made possible:

- **Shot creation:** mostly the share of twos and threes that weren't assisted, with some usage rate
  mixed in. Putbacks also count as unassisted, so without usage a rebounding big would look like a
  shot creator.
- **Rim pressure:** share of shots at the rim, shooting fouls drawn and and-ones.
- **Rim finishing** (rim runners and lob threats): dunk share, finishing at the rim and how often
  those baskets were assisted.
- **Spot-up shooting:** assisted threes, three-point percentage, corner share and three-point rate.
- **Post and mid-range scoring:** shot share and accuracy from those distances.

A few of these are still stand-ins, and the UI labels them. There is no play-type data, so dunks
and assisted rim finishes stand in for rolls and lobs, and shots from 3 to 10 feet include
floaters as well as post-ups. Steals, blocks and charges drawn measure defensive activity, not
overall defensive impact.

Players with no style text at all either get no style credit (the default) or the median style
score. Now that everyone has a stat-based profile, this only matters if you rebuild the index
without them.

Hard filters always run in SQL before any ranking happens, so a strong style match can never sneak
in a player who fails a filter.

### Explanations

Each result gets an evidence package: formatted stats, percentiles, the score breakdown and the
passages with their IDs. The LLM writes from that package and nothing else. Afterwards, a grounding
check throws out any explanation that states a number not found in the player's evidence, cites a
passage it wasn't given, or calls someone "the best". When that happens, the template explanation
is shown instead and the UI says why.

## Evaluation

The point of the project is to check whether combining stats and descriptions actually beats either
one alone, so the evaluation compares three methods on the same questions:

- **Stats only:** SQL filters plus statistical ranking.
- **Profiles only:** plain vector search over the style profiles. It deliberately ignores the SQL
  filters, the way a basic vector search would.
- **Hybrid:** both, combined as described above.

There are 13 test questions in `eval/queries.json`. They cover efficiency and playmaking,
scoring-first players, low-usage playmakers, "similar to" questions, explicit number filters,
style-heavy questions, a question the data can barely support, and one it can't support at all.

Grading follows the pooling approach used in information-retrieval research. The top 15 results
from every method are merged into one list per question, and each player is graded by hand: 0 for
not relevant, 1 for partly relevant, 2 for highly relevant.

```bash
python scripts/evaluate.py pool   # writes eval/label_pool.csv with an empty grade column
# grade the players, save the result as eval/labels.csv
python scripts/evaluate.py run    # writes eval/results.md and eval/results.json
```

The report includes Recall@10 (how many of the relevant players show up in the top 10), nDCG@10
(whether the most relevant players are near the top), judged@10 (how much of the top 10 was
actually graded) and median latency per stage.

**There are no results yet**, because the labels haven't been graded. Even once they are, a dozen
questions graded by one person is a small sample, so I'll treat the numbers as a sanity check
rather than proof that one method wins.

On latency so far: SQL and ranking take a few milliseconds, and the vector search takes a few
hundred. The first search after startup takes several seconds while the embedding model loads.

One thing to keep in mind when reading the results: the profiles are built from the same stats the
ranking uses, so the interesting question is whether the hybrid method beats stats only at all.
What the style search can add is matching by meaning, for example finding "lob threats" through
words like "dunk rate" and "assisted twos" without anyone mapping those words to columns by hand.

## Tests

`pytest` runs 78 tests against small synthetic fixtures: obviously fake players and a simple
hashing embedder, so the suite runs in under a second without downloading anything. They cover:

- schema validation and the rule against invented thresholds
- filter extraction and SQL filter enforcement, including how missing values are handled
- hard filters always beating style matches
- the percentile and direction math, including stats that count in reverse for one goal
- scaling style scores separately for each type of text, and both missing-description policies
- the stat-based profile template, its style tag rules, and the number-free summary that gets embedded
- the importer's unit conversions
- "similar to" searches and empty-result messages
- the grounding check catching invented stats and fake citations
- fallbacks when an LLM fails or returns bad output, and Gemini going first with Ollama as backup (all mocked)
- Gemini retrying overloaded and rate-limited requests, checked against a fake Gemini server on localhost
- the evaluation metrics

## Known limitations

- Only one season is loaded, and the schema is built for more but nothing else is imported yet.
- All style text is built from the same stats used for ranking, so there is no independent view of
  how anyone plays (scouting reports, film notes).
- The style tag thresholds in the profiles are my judgment calls and haven't been validated.
- There's no play-type tracking data, so rolls, lobs, post-ups and defense rely on stand-in stats.
- The grounding check confirms that a number appears somewhere in a player's evidence. It doesn't
  confirm the number was attached to the right stat.
- The rule-based parser works from keywords, so unusual phrasings can slip past it.
- Comparisons in filters are inclusive, so "over 20 points" is treated as 20 or more.
- I've checked the Gemini path on real questions, but I haven't benchmarked a specific local model
  yet.
- The ranking weights haven't been tuned.

## Project layout

```text
app.py                          Streamlit UI
config.py                       settings, all overridable with environment variables
scripts/import_stats.py         loads Basketball-Reference stats into SQLite
scripts/build_stat_profiles.py  writes a stat-based style profile for every player
scripts/build_index.py          embeds passages and builds the FAISS index
scripts/evaluate.py             builds the grading pool and computes the metrics
src/schemas.py                  search plan, metric list, ranking goals
src/query_parser.py             LLM parser and rule-based fallback
src/llm/                        Gemini, Ollama and Anthropic adapters, prompts
src/stats/                      database schema, parameterized queries, stat-based profile template
src/retrieval/                  style index, ranking math, the three search methods
src/generation/                 evidence packages, explanations, grounding check
src/pipeline.py                 ties parsing, search and explanation together
eval/                           test questions, labels, results
tests/                          pytest suite
```

# FullCourt: an NBA research assistant

FullCourt answers questions like "who are the efficient scoring guards that can also run an offense?"
by combining two kinds of evidence: hard numbers from a stats database, and written descriptions of
how players actually play. Every number you see comes from the database, and every claim about
playing style links back to the passage it came from.

I built it as a portfolio project to explore hybrid retrieval, meaning SQL and vector search working
together, and to see how far you can keep an LLM honest by only letting it talk about evidence it
was handed. It covers one season (2025-26) and doesn't train any models.

## How a question flows through the app

1. **Parsing.** The question is turned into a structured search plan: season, positions, explicit
   numeric filters, ranking preferences and a short description of the playing style to look for.
   A local LLM does this when one is available. Otherwise a rule-based parser takes over.
2. **Stats lookup.** SQL applies the filters and pulls each candidate's numbers.
3. **Style search.** The style description is embedded and compared against player-profile passages
   stored in a FAISS index.
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
python scripts/import_descriptions.py   # pulls Wikipedia player-profile sections (about 5 minutes, cached)
python scripts/build_index.py           # embeds the passages and builds the FAISS index

streamlit run app.py
pytest
```

I've tested this with Python 3.14 on an M2 Mac. The first run downloads the embedding model (about
130 MB) from Hugging Face.

### Using a local LLM

The LLM is optional and runs through [Ollama](https://ollama.com). Pull a model, copy `.env.example`
to `.env`, and set `FULLCOURT_LLM_MODEL` to the model's name. The app asks Ollama for output that
matches a JSON schema and requests an 8192-token context window, since Ollama's default is too small
for the explanation prompt.

Choose a model that fits comfortably in memory alongside the app. On an 8 GB machine, a 7B or 8B
model will swap heavily and take minutes per question; a 3B or 4B model is a better fit.

If Ollama isn't running or the model isn't pulled, nothing breaks. The app switches to the
rule-based parser and template explanations and tells you so in the sidebar. There's also an
Anthropic adapter if you'd rather use an API (`FULLCOURT_LLM_PROVIDER=anthropic`, plus
`pip install anthropic` and an API key).

## Where the data comes from

**Stats** come from Basketball-Reference's
[per-game](https://www.basketball-reference.com/leagues/NBA_2026_per_game.html) and advanced tables
for the 2025-26 regular season: 582 players in all. When a player was traded, I keep their combined
season row and record each team they played for.

**Playing-style descriptions** come from the "Player profile" (or "Playing style") sections of
English Wikipedia articles, which are licensed CC BY-SA 4.0. Each passage keeps a link to the exact
article revision it was taken from, along with the revision date.

None of the descriptions are written by a model. If a player doesn't have a profile section, the
app says there's no style evidence for them instead of making something up.

A few practical notes:

- Percentages and rates are stored as fractions, so a 61.6% true shooting mark is `0.616`.
  Basketball-Reference reports usage, assist, turnover, rebound, steal and block rates in percentage
  points, so the import divides those by 100.
- Missing values stay `NULL`. They're never filled in with zero.
- The scraped pages, the database and the index aren't committed. The scripts rebuild them.
  Basketball-Reference data is used here for a non-commercial demo, so check their terms before
  redistributing it.

### Description coverage

This is the biggest weakness of the current dataset. Of the 406 players who meet the default
eligibility bar (at least 20 games and 10 minutes per game), only **48 have a Wikipedia profile
section**, which works out to 110 passages. Another 347 have articles without one, and 11 couldn't
be matched to an article with confidence (mostly names with initials, like P.J. Washington). The
players who do have profiles are mostly stars, so style evidence is thin for role players.

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

Passages are embedded with `BAAI/bge-small-en-v1.5` and stored in a FAISS inner-product index, so
scores are cosine similarities. A player's style score is the similarity of their best-matching
passage. For "players similar to X" questions, the search uses the average of X's own passage
vectors instead of the question text.

### Ranking

Each preference maps to a fixed set of stats. Playmaking, for example, is half assist rate and half
assists per game. When two stats measure the same thing, they're averaged rather than counted
twice.

Stats are converted to percentiles within the filtered pool of candidates, so different scales
become comparable and outliers don't dominate. Turnover rate is flipped so that lower counts as
better. The statistical fit is the average of the preference scores. Style scores are scaled from 0
to 1 across the pool, and the final score is 0.6 times the statistical fit plus 0.4 times the style
score. Those weights are a starting point, not something I've tuned.

Players without a description either get no style credit (the default) or the median style score.
With coverage this low, that choice changes results a lot.

Two of the stats are stand-ins, and the UI labels them as such. Usage rate stands in for shot
creation, because the dataset has no data on unassisted shots. Steal and block rates stand in for
defense, which they only partly capture.

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
- **Descriptions only:** plain vector search over the passages. It deliberately ignores the SQL
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

## Tests

`pytest` runs 47 tests against small synthetic fixtures: obviously fake players and a simple
hashing embedder, so the suite runs in under a second without downloading anything. They cover:

- schema validation and the rule against invented thresholds
- filter extraction and SQL filter enforcement, including how missing values are handled
- hard filters always beating style matches
- the percentile and direction math, and both missing-description policies
- "similar to" searches and empty-result messages
- the grounding check catching invented stats and fake citations
- fallbacks when the LLM fails or returns bad output, using a mocked Ollama
- the evaluation metrics

## Known limitations

- Only one season is loaded, and the schema is built for more but nothing else is imported yet.
- Style coverage is thin and skewed toward stars (see above), which biases the hybrid ranking toward
  players who happen to have a Wikipedia profile.
- Wikipedia profiles describe a player's whole career, so they can be out of date for 2025-26.
- Shot creation and defense rely on stand-in stats.
- The grounding check confirms that a number appears somewhere in a player's evidence. It doesn't
  confirm the number was attached to the right stat.
- The rule-based parser works from keywords, so unusual phrasings can slip past it.
- Comparisons in filters are inclusive, so "over 20 points" is treated as 20 or more.
- I've tested the LLM path with mocked responses but haven't benchmarked a specific local model
  yet.
- The ranking weights haven't been tuned.

## Project layout

```text
app.py                          Streamlit UI
config.py                       settings, all overridable with environment variables
scripts/import_stats.py         loads Basketball-Reference stats into SQLite
scripts/import_descriptions.py  loads Wikipedia profile sections into SQLite
scripts/build_index.py          embeds passages and builds the FAISS index
scripts/evaluate.py             builds the grading pool and computes the metrics
src/schemas.py                  search plan, metric list, ranking goals
src/query_parser.py             LLM parser and rule-based fallback
src/llm/                        Ollama and Anthropic adapters, prompts
src/stats/                      database schema and parameterized queries
src/retrieval/                  style index, ranking math, the three search methods
src/generation/                 evidence packages, explanations, grounding check
src/pipeline.py                 ties parsing, search and explanation together
eval/                           test questions, labels, results
tests/                          pytest suite
```

# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **post** | One text of the corpus | statement (outside the CSV column), message |
| **label** | One of the 7 categories of the corpus | class name, diagnosis, status (outside the CSV column) |
| **corpus** | The full table of posts and labels | dataset (for the corpus), data dump |
| **source** | The original dataset or platform of a post | origin, provider |
| **duplicate group** | Posts that are copies or near copies of each other | cluster, family |
| **split** | The train, validation or test part of the corpus | fold, partition |
| **prompt template** | The one text frame around a post (`classify-v1`) | prompt format, instruction |
| **completion** | A space and the label text after the prompt | answer, output |
| **label scoring** | The sum of the token log-probabilities of a completion | generation, decoding |
| **temperature** | The calibration divisor of the log-scores | softmax temperature (in sampling) |
| **routing rule** | P(Suicidal) at or above the threshold, or a crisis phrase | alarm, filter |
| **crisis phrase** | A text pattern in `safety.py` that always routes a post | keyword, trigger word |
| **macro-F1** | The mean of the F1 scores of all labels | accuracy (for macro-F1), score |
| **ECE** | Expected calibration error with 15 bins | calibration score |
| **model folder** | `model.json` plus `model.joblib` or a LoRA adapter | checkpoint, artefact |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read the corpus into the common schema |
| **clean** | Replace URLs and mentions, remove HTML, collapse white space |
| **group** | Put copies and near copies into one duplicate group |
| **split** | Make the stratified, group-aware train, validation and test parts |
| **fit** | Train a model on the train split |
| **calibrate** | Fit the temperature on the validation split |
| **route** | Send a post to a human reviewer by the routing rule |
| **evaluate** | Calculate the test metrics, slices and routing results |

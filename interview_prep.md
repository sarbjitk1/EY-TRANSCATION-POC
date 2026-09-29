# EY Interview Prep: Questions & Answers

> Private prep notes. Not part of the submission (git-ignored).
> Answer formula for every question: **direct answer → evidence with a number → caveat → business link.**
> Speak slowly. Pause silently instead of saying "uh". End every answer with a clear final sentence.

---

## 0. Numbers to know by heart

| What | Value |
|---|---|
| Raw rows | 5,899 |
| Exact duplicates removed | 679 → **5,220** rows |
| Train / hold-out | **4,698 / 522** (stratified 90/10, seed 42) |
| Classes | 6 (11 raw label spellings) |
| Class sizes (train / hold-out) | C1 4,189/466 · C2 461/51 · C3 21/2 · C4 14/2 · C5 2/0 · C6 11/1 |
| Majority baseline accuracy | **89.3%** (macro-F1 ≈ 0.19) |
| Final model | Logistic Regression, `class_weight="balanced"`, C=10 |
| Model features | **6,217** (Col1 TF-IDF 1,165 · Col4 4,371 · Col6 670 · Col7 one-hot 4 · numeric 7) |
| CV macro-F1 (5-fold) | LR **0.784 ± 0.06** · RF 0.665 · HGB 0.603 · Ensemble 0.646 |
| CV accuracy | LR 0.946 · RF 0.950 · HGB 0.935 · Ensemble 0.949 |
| C tuning (macro-F1) | C=1 → 0.700 · **C=10 → 0.784** · C=50 → 0.765 |
| Hold-out accuracy | **96.4%** |
| Hold-out macro-F1 | **0.898** |
| Hold-out balanced accuracy | **0.981** |
| Hold-out errors | 15 C1→C2, 1 C1→C6, 3 C2→C1 (19 wrong of 522) |
| Per-class recall | C1 96.6% · C2 94.1% · C3 2/2 · C4 2/2 · C6 1/1 |
| Category_2 precision | 48/63 ≈ 76% (cost of balanced weights) |
| Column importance (permutation, macro-F1 drop) | Col1 0.26 · Col4 0.25 · Col6 0.05 · Col3 0.01 · Col7 ~0 · Col2/Col5 0 (not used) |
| Dropping Col2 + Col5 | CV macro-F1 **0.743 → 0.784** |
| Pipeline time for 522 rows | ~0.09 s |
| Low-confidence rows (<70%) | 21 of 522 |
| Hold-out vendors also seen in train | 97.9% |
| Conflicting duplicates (same features, different label) | 1 pair (C1 vs C2) |

---

## 1. Opening / big picture

**Q1. Give us a 60-second summary of your project.**
The client has about 5,900 accounting transactions and wants a 6-class label predicted from seven columns. It looks easy but isn't: 89% of rows are Category_1, so a model that always says Category_1 already scores 89% accuracy. The labels also had 11 spellings, there were 679 exact duplicates, and four classes have fewer than 25 examples. I cleaned the labels, removed duplicates before splitting, and chose models by macro-F1 instead of accuracy. Logistic regression on TF-IDF features won clearly, with CV macro-F1 0.78 against 0.60–0.67 for tree models. On the untouched hold-out it scores 96.4% accuracy and 0.98 balanced accuracy. The whole pipeline is a single sklearn object, so Tab 2 runs exactly the training transformations live, and each prediction can be explained on demand by an AI agent that uses tools.

**Q2. What was the hardest part?**
The class imbalance. Categories 3–6 have 2 to 21 training rows each, so every decision was shaped by it: the metric (macro-F1), the split (stratified), validation (5-fold CV instead of one validation set), and the model (balanced class weights). Category_5 has only 2 rows in total, so I can't evaluate it at all, and I say so openly.

**Q3. What would you do differently with more time?**
1. Split by vendor (GroupKFold on Col1) to measure performance on vendors the model has never seen. Today 98% of hold-out vendors also appear in training.
2. Nested CV for an unbiased estimate after tuning.
3. Wider tuning: n-gram range, min_df, per-class weights, decision thresholds.
4. Probability calibration so the confidence score means what it says.
5. A human-review queue for low-confidence rows, plus drift monitoring in production.

**Q4. What did you use GenAI for, and how do you know the code is right?**
I used it to scaffold the Streamlit UI and speed up boilerplate. Every modelling decision I made and checked myself: the leakage-safe order (dedupe → split → fit inside CV), the metric choice, and the feature choices, each validated against CV scores. I can walk through any line.

**Q5. Why did you pick Streamlit?**
It's Python end to end, so the dashboard imports the same `src/pipeline.py` used in training, with no second implementation in JavaScript that could drift out of sync. It's quick to build interactive charts, file upload and session state. For a production client app I'd put the model behind a FastAPI service and use a separate frontend.

---

## 2. Data exploration

**Q6. What did you find in the data?**
Five main issues: (1) 11 spellings of the label, such as `Category 3`, `Categry_6` and `category_1`, which all map to 6 real classes; (2) 679 exact duplicate rows; (3) severe imbalance, with Category_1 at about 89%; (4) Col2 IDs are almost all unique and some were corrupted by Excel into scientific notation like `4.80Z+11`; (5) 153 missing Col4 values, amounts stored as text with commas, and some negative amounts (credits).

**Q7. What are the columns?**
The data is anonymised. From the patterns I interpreted them as: Col1 vendor/payee (word tokens), Col2 document/invoice ID, Col3 amount, Col4 line description (word tokens), Col5 posting date, Col6 account/GL description (word tokens), Col7 document type. These are interpretations; the model doesn't depend on them.

**Q8. How did you check correlations when most columns are text?**
The usual correlation matrix only covers numeric columns. For the text and categorical columns I looked at the class distribution per value in the dashboard, and I used permutation importance on the trained model, which shows Col1 and Col4 carry most of the signal.

**Q9. What about outliers?**
Amounts are heavy-tailed. I kept outliers because they are real transactions, not data errors, and removing large invoices would bias the model. Instead I used a signed log transform, `sign(x)·log(1+|x|)`, which compresses the scale while keeping the sign for credits.

**Q10. How did you handle missing values?**
Missing text becomes an empty string, which means no tokens. I also added a `col4_missing` flag in case missingness itself is informative. Missing or unparseable numbers are filled with the training median inside the pipeline, and Col7 gets its own "<missing>" category.

**Q11. Were there any conflicting labels?**
Yes, exactly one pair: identical features, one labelled Category_1 and one Category_2. That's label noise and no model can get both right. I'd raise it with the client rather than guess.

**Q12. Was anything surprising?**
Col2 (the ID) and Col5 (the date) looked useful but weren't. When I removed them, CV macro-F1 rose from 0.743 to 0.784. They mostly added noise that hurt the rare classes.

---

## 3. Cleaning & preprocessing

**Q13. How did you clean the labels?**
With a regex that takes the number after "cat…", after converting words to digits ("three" → 3), so `Category _3`, `Categry_6` and `category_1` all become `Category_<n>`. Labels it can't read are reported and removed, never guessed. The same function also cleans an optional label column in an uploaded file, so live accuracy works.

**Q14. Why remove duplicates, and why before the split?**
If a row and its exact copy end up on opposite sides of the split, the test measures memorisation, not generalisation. Removing duplicates first guarantees no row appears in both train and test.

**Q15. Couldn't the duplicates be real, like two identical invoices?**
Possibly. For the model it doesn't matter: a duplicate adds no new information and only overweights that pattern. For the client I'd flag possible double postings as a separate finding, which is useful to an audit team.

**Q16. Why TF-IDF for the text columns?**
The tokens are anonymised IDs like `Word563`, so pretrained embeddings or language models can't recognise any meaning. What carries signal is which tokens and token pairs appear together. TF-IDF with unigrams and bigrams captures that, down-weights very common tokens, and is sparse and fast. I split on whitespace and turned off lowercasing because the tokens are IDs, and used sublinear TF to damp repeated tokens.

**Q17. Why bigrams?**
A vendor name like "Word42 Word43" is more specific as a pair than as two separate words. Bigrams let the linear model learn those combinations. For the tree models I used unigrams with a capped vocabulary for speed.

**Q18. Why didn't you use word embeddings or BERT?**
The words are anonymised IDs, so pretrained models have no knowledge of them. Training embeddings from scratch on about 4,700 rows would overfit. TF-IDF is the right tool for this data size and type.

**Q19. What did you do with the amount?**
Parsed text like `1,092.50` and `(45.00)` into floats. Then a signed log, a negative flag (credits) and a round-number flag (amounts divisible by 100 often mean estimates or accruals). Then standard scaling, because logistic regression is sensitive to feature scale.

**Q19b. What is `amount_is_round`, and why multiples of 100 and not 10?**
It's 1 when the amount is an exact multiple of 100 (600, 4,100), a heuristic for manually entered amounts such as accruals or fixed fees, while real invoices look like 3,766.80. I chose 100 because multiples of 10 are very common by chance (36% of all amounts). When I later checked other thresholds, I found 83% of Category_3 amounts are multiples of 10, against 36% overall. With only 23 rows that could be coincidence, so the next step would be to add ÷10 and whole-number flags and test them in cross-validation. Overall the amount contributes little (permutation drop ≈ 0.01).

**Q20. What did you do with Col2 (the ID)?**
I engineered its *shape*: `KBNZ072618` becomes `A9`, so the model could learn a vendor's numbering format without memorising individual IDs. I also flagged the Excel scientific-notation corruption. It was a good idea but didn't help: removing Col2 and Col5 improved CV macro-F1, so the final model doesn't use them. I kept them in the EDA.

**Q21. Why not one-hot encode the ID?**
It's almost unique per row, so one-hot would give thousands of columns seen only once. That's pure memorisation and useless for new data.

**Q22. What did you do with the date?**
I engineered month, day, and month-start/month-end flags, since month-end often means accruals. It didn't improve CV, so it's not in the final model.

**Q23. How is Col7 encoded?**
One-hot with `min_frequency=2`, so values seen only once are grouped together, and `handle_unknown="ignore"`, so a new document type at inference doesn't crash the pipeline.

**Q24. How do you prevent preprocessing leakage?**
All preprocessing (TF-IDF, imputer, scaler, encoder) lives inside one sklearn `Pipeline` together with the classifier. During cross-validation the whole pipeline is refit on each training fold, so the vocabulary, medians and scaling values never see the validation fold.

**Q25. Why scale features for logistic regression?**
Regularisation penalises all coefficients equally. Without scaling, a feature measured in large units would be penalised differently from a 0/1 flag. Scaling puts them on equal footing and helps the solver converge.

**Q26. Show us before/after.**
Tab 1 → Cleaning & preprocessing shows raw rows next to the engineered features and the final width of each feature block.

---

## 4. Split & validation strategy

**Q27. How did you split the data?**
90% train / 10% hold-out, stratified by label, seed 42, after removing duplicates. The hold-out was used exactly once, at the very end. Inside the 90% I used 5-fold stratified cross-validation for both model comparison and tuning.

**Q28. Why no separate validation set?**
A fixed 10% validation set would contain only 1–2 examples of each rare class, so the score would swing with one prediction. With 5-fold CV every training row is used for validation once, which gives a much more stable estimate plus a standard deviation.

**Q29. Why stratify?**
Without stratification, a random split could put zero Category_3 or Category_4 rows in the test set. Stratifying keeps class proportions the same in both parts.

**Q30. Why 5 folds and not 10?**
Category_4 has 14 training rows, so 5 folds gives about 3 per validation fold. 10 folds would give 1–2, which is too noisy. It's also twice as fast.

**Q31. Category_5 has 2 rows. How did stratification handle it?**
Both ended up in training, so the hold-out has none and I can't evaluate that class. The code warns and sets aside any class with fewer than 2 rows instead of crashing. In practice I'd ask the client for more examples or send those predictions to human review.

**Q32. Isn't selecting on CV and tuning on the same folds optimistic?**
Slightly, yes. That's why the untouched hold-out exists as the final check. The gold standard is nested CV, which I'd add next. The tuning grid was small (3 values of C), so the optimism is limited.

**Q33. Would a random split be realistic for a client?**
Not completely. 98% of hold-out vendors also appear in training, so the 96% mostly reflects known vendors. For new vendors I'd expect lower performance. A GroupKFold by vendor, or a time-based split if the dates are reliable, would test that. It's my first next step.

---

## 5. Model selection & tuning

**Q34. Which models did you try?**
Four, all with balanced class weights and identical preprocessing inside CV: Logistic Regression, Random Forest, Histogram Gradient Boosting, and a soft-voting ensemble of all three.

**Q35. Why did logistic regression beat the tree models?**
Three reasons. (1) The feature space is about 6,200 sparse TF-IDF columns and there are only about 4,700 rows. Linear models are the classic strong baseline for high-dimensional sparse text. (2) Tree models split on one feature at a time and struggle with sparse text; I had to cap their vocabulary at 250–500 terms to make them run, which discards signal. (3) In a regularised linear model, balanced class weights give rare classes real influence, while trees with 11–21 examples overfit or ignore them. The CV results confirm it: 0.78 macro-F1 against 0.60–0.67.

**Q36. Are you sure you didn't just under-tune the trees?**
It's a fair challenge. I did tune them (RF: trees and max_features; HGB: learning rate and leaf count), but they were limited by the vocabulary cap needed for speed. Still, the gap is about 0.12 macro-F1, bigger than tuning usually closes, and the standard deviation across folds is higher for the trees (0.10–0.13 against 0.06), so they're also less stable. With more time I'd try gradient boosting on SVD-reduced TF-IDF, or a linear SVM.

**Q37. Why not XGBoost / LightGBM?**
LightGBM needs the system `libomp` library on Mac, which is a fragile dependency for a live laptop demo. sklearn's `HistGradientBoostingClassifier` is the same algorithm family, and it lost clearly. I'd expect XGBoost to perform about the same.

**Q38. Why did the ensemble lose?**
Averaging with two weaker models diluted the logistic regression's rare-class predictions. Accuracy rose slightly (0.949) but macro-F1 dropped to 0.646. An ensemble only helps when its members are similarly strong and make different mistakes.

**Q39. Why macro-F1 as the selection metric?**
It gives equal weight to all classes. Accuracy is dominated by Category_1: always predicting C1 scores 89% accuracy but macro-F1 of only about 0.19. The client presumably cares about finding the rare categories, and macro-F1 rewards exactly that.

**Q40. What does C mean and how did you tune it?**
C is the inverse of regularisation strength: higher C means less regularisation. I grid-searched C = 1, 10 and 50 with 5-fold CV. C=1 was too constrained (0.70 macro-F1) because the rare classes need room, C=50 started to overfit (0.765), and C=10 was best (0.784).

**Q41. What does class_weight="balanced" do?**
It weights each class's loss by the inverse of its frequency, so misclassifying one Category_4 row costs about 300 times as much as misclassifying a Category_1 row. Without it the model would mostly ignore the rare classes.

**Q42. Why not SMOTE or oversampling?**
SMOTE interpolates between feature vectors, and interpolating sparse TF-IDF vectors produces meaningless synthetic "documents". With only 2–21 examples, oversampling mostly copies the same rows. Class weighting achieves the same rebalancing without creating fake data or any leakage risk.

**Q43. Why not a neural network?**
About 4,700 rows of anonymised tokens is too little data. A network would overfit, be slower, and be harder to explain. Logistic regression is accurate here, trains in 2 seconds, and is transparent, which matters for accounting and audit clients.

**Q44. Is logistic regression multiclass here?**
Yes, sklearn uses multinomial (softmax) logistic regression, so it outputs a probability for each of the 6 classes, adding up to 1. That's what the confidence score and per-class probabilities in Tab 2 show.

**Q45. Did you try feature selection?**
Yes, at the column level. Removing Col2 and Col5 improved macro-F1. Within the text, L2 regularisation handles the thousands of TF-IDF features. Explicit selection like chi-squared would be a reasonable next experiment.

**Q46. How long does training take?**
Logistic regression cross-validates in about 2 seconds. The full script (4 models × 5-fold CV + grid search + evaluation) runs in about 90 seconds.

---

## 6. Results & evaluation

**Q47. Your accuracy is 96%, but 89% is the baseline. Is the model actually good?**
Yes. Always predicting C1 gives 89% accuracy but macro-F1 of about 0.19. My model scores macro-F1 0.90 and balanced accuracy 0.98 on the hold-out. Per class, recall is 97% for C1 and 94% for C2, and every C3, C4 and C6 hold-out row was correct. Caveat: those rare classes have only 1–2 hold-out rows, so I treat CV macro-F1 of 0.78 as the realistic estimate.

**Q48. Why is hold-out macro-F1 (0.90) higher than CV (0.78)?**
Small-sample noise. The hold-out has only 2, 2 and 1 rows for C3, C4 and C6, so getting all of them right pushes their F1 to 1.0. CV averages over 5 folds and many more rare-class rows, so it's the more reliable number. I wouldn't promise 0.90 to a client.

**Q49. Walk me through the confusion matrix.**
Of 522 rows, 19 are wrong. 15 are Category_1 predicted as Category_2, 1 is C1 predicted as C6, and 3 are C2 predicted as C1. Categories 3, 4 and 6 are all correct. So the main confusion is between C1 and C2, and it leans toward flagging C2.

**Q50. Why does the model over-predict Category_2?**
That's the balanced class weight at work. It trades some C1 precision for high recall on the smaller classes. C2 precision is about 76% and recall 94%. If a missed C2 is costlier than a false alarm (e.g. a flagged transaction someone reviews), this is the right trade-off. If not, I can raise the decision threshold for C2.

**Q51. Precision vs recall: which matters more here?**
It depends on the client's cost of each mistake, and I'd ask them. For something like transaction classification in audit, missing a rare, important category is usually worse than an extra review, so I leaned toward recall. The trade-off can be adjusted with per-class thresholds without retraining.

**Q52. Which features matter most?**
Permutation importance on the hold-out: shuffling Col1 (vendor) drops macro-F1 by 0.26 and Col4 (line description) by 0.25, then Col6 (account) by 0.05. Amount and document type contribute little. That makes business sense: who you paid and what for says more about the category than how much.

**Q53. Why permutation importance instead of the coefficients?**
There are 6,217 coefficients per class spread across TF-IDF terms, which doesn't answer "which column matters". Permutation importance works at the raw-column level, is model-agnostic, and measures the actual drop in the metric I care about.

**Q54. How confident are you the model will hold up in production?**
Confident for known vendors and common classes. Less so for new vendors (98% of test vendors were seen in training) and for the rare classes, which have little support. I'd deploy with a confidence threshold that sends uncertain rows to a human, and monitor for drift.

**Q55. What is balanced accuracy?**
The average recall across classes. It's 0.98 here, meaning the model finds nearly all rows of every class, not just the big one.

---

## 7. Tab 2: pipeline & engineering

**Q56. What happens step by step when I upload a file?**
Five timed steps, all live: (1) **Read:** CSV or Excel, everything as text so IDs aren't auto-converted, delimiter detected automatically. (2) **Schema:** `coerce_schema` maps Col1–Col7 or A–G headers (any case or spacing) to the expected columns; a missing column is filled with blanks and a warning is shown. (3) **Feature engineering:** parse amounts, signed log, token counts, flags. (4) **Encode and score:** TF-IDF, one-hot and scaling, then logistic regression. (5) **Display:** predicted class, confidence, and every class probability in a scrollable table.

**Q57. How do you guarantee the same preprocessing as training?**
Steps 3 and 4 are not re-implemented in the app. They're one sklearn Pipeline object, fitted during training and saved to `model.joblib`. The app loads it and calls `predict_proba`, so it's the same code with the same fitted vocabulary and scaling values, and nothing is refit on uploaded data. **One fitted Pipeline object, saved once, loaded at inference, never refit.**

**Q58. What edge cases does the upload handle?**
- CSV or Excel, any delimiter, bad encoding characters replaced
- Headers Col1–Col7 or A–G, any case or spacing; no header row → first row treated as data
- Unrecognised headers → first 7 columns mapped by position, with a warning
- Missing columns → filled with blanks, with a warning
- Blank rows dropped; an empty file gives a clear error
- Unparseable amounts or dates → counted, shown, and imputed
- Unseen vendors, words or document types → ignored safely
- Optional label column → live accuracy shown
- Any other failure → an error message in the UI, not a crash

**Q59. What if someone uploads a file with extra columns, or in a different order?**
Columns are matched by name, so order doesn't matter and extra columns are ignored. If there's also a label column, it's used to show live accuracy.

**Q60. What if a column is completely empty?**
The pipeline still runs. Text becomes empty, and numbers are filled with the training median. Predictions will be less confident, which the confidence column and the low-confidence counter show.

**Q61. What is the "low-confidence rows" metric?**
Rows where the model's highest probability is below 70%. There are 21 of 522 in the hold-out. The toggle filters the table to just those. In production they'd go to a human reviewer instead of being auto-classified.

**Q62. Does the pipeline re-run if I interact with the page?**
No. The uploaded file is hashed (MD5), so the pipeline runs once per new file and the results are kept in session state. Clicking Explain or filtering doesn't re-run predictions. Upload a different file and it runs fresh.

**Q63. How fast is it? Would it scale?**
About 0.09 seconds for 522 rows. Everything is vectorised, so tens of thousands of rows are fine. For millions I'd run batch scoring outside the UI or put the model behind an API with chunked processing.

**Q64. How would you deploy this for a real client?**
Package the pipeline in a FastAPI service in Docker, with a version tag on each model and schema validation at the endpoint. Keep the dashboard as the front end, log inputs and predictions for monitoring, and retrain on a schedule or when drift is detected. The API key would live in a secrets manager, not a file.

**Q65. How would you monitor the model in production?**
Track input drift (new vendors, amount distribution, share of unknown tokens), prediction drift (class mix, share of low-confidence rows), and actual performance where reviewers correct labels. Alert when any of these moves past a threshold, then retrain.

**Q66. How do you version the model?**
Right now `model.joblib` together with `metrics.json` and the training log from the same run, and pinned library versions in `requirements.txt`. In production I'd use MLflow or a model registry, tied to the data snapshot and git commit.

**Q67. Why is everything read as a string?**
So pandas doesn't guess types: IDs could turn into floats or scientific notation, and "1,092.50" would fail to parse. My own parsers then convert amounts and dates deliberately, the same way in training and at upload.

---

## 8. The AI explainer (agent)

**Q68. How does the explainer work?**
When you click Explain on a row, an LLM agent (gpt-4o-mini) receives that row and has three tools: (1) **feature contributions:** occlusion-based attributions per column and per word; (2) **similar training records:** nearest labelled neighbours in the model's feature space; (3) **class profile:** typical vendors, accounts and amounts for a class. It must call the contribution tool, decides for itself whether it needs the others, and then writes a short explanation. The tool calls are shown in the UI.

**Q69. What makes it "agentic" rather than a single prompt?**
The model chooses which tools to call and in what order, based on what it sees. For example, it looks up the runner-up class profile when confidence is low, or checks neighbours for a rare class. It loops, calling tools and reading their results, until it can answer, with a limit of 6 steps.

**Q70. What is occlusion, and how is it calculated?**
For each input, I replace it with a neutral value (blank text, median amount, most common document type), re-score the row, and measure how much the predicted class's probability drops. A large drop means that input supported the prediction. The same is done for each word in the text columns. All variants are scored in a single batch, so it's fast.

**Q71. Why occlusion and not SHAP?**
SHAP on a pipeline where text becomes 6,000 TF-IDF columns gives attributions per TF-IDF term, not per business column, and needs extra setup. Occlusion works directly on raw columns and words, works with any model, and is easy to explain to a client. Its limitations: the result depends on the chosen neutral value, and it misses interactions between features. LinearSHAP would be a good cross-check.

**Q72. How do you stop the LLM from hallucinating?**
(1) The numbers come from deterministic tools, not from the LLM. (2) The system prompt requires it to use the tool's `effect` field for direction and not to invent meanings for anonymised tokens. (3) Temperature 0.2. (4) A step limit. (5) The tool trace is shown, so a user can check every claim. (6) Tool errors are passed back to the model instead of crashing.

**Q73. Why only on demand?**
The brief requires it, and it's sensible anyway: explaining 522 rows up front would cost 522 LLM calls, most of which nobody would read. Each explanation is cached per row, so clicking the same row again is instant.

**Q74. What if there's no internet or no API key?**
It falls back to a rule-based explanation built from the same tools (contributions and neighbours), clearly labelled as the fallback. The demo still works offline.

**Q75. Why gpt-4o-mini?**
It's fast and cheap, and good enough because the tools do the analysis; the LLM only chooses tools and writes the text. It can be changed with the `OPENAI_MODEL` environment variable. For a client with data-residency concerns I'd use Azure OpenAI or a local model.

**Q76. Is it safe to send client data to OpenAI?**
For a real client I'd check that first. Options: Azure OpenAI with a data-processing agreement and no training on client data, a locally hosted model, or sending only the tool outputs (contributions and token IDs) and never raw values. This data is already anonymised.

**Q77. How are the "similar records" found?**
The row is transformed with the model's own fitted preprocessing (TF-IDF and friends), and I find the nearest training rows by cosine similarity. It's precedent-based evidence: "4 of 5 similar historical records were labelled Category_1".

**Q78. Can the explanation contradict the model?**
The prompt forbids claiming an input pushes toward a class unless the tool output shows it. The neighbour votes can disagree with the prediction, and that's shown deliberately as a reliability warning, not hidden.

---

## 9. Business & communication

**Q79. How would you explain this model to a non-technical client?**
"The model reads who the payment went to, what the line says and which account it hit, and compares that to thousands of past labelled transactions. It gets about 96 in 100 right, and it tells you when it's unsure so a person can check those. For any row you can click Explain to see why."

**Q80. What business value does this bring?**
Automatic labelling of the roughly 96% of transactions it's confident about, with humans focused on the uncertain ones. That means faster close or review cycles, consistent labels, and an audit trail of why each label was given.

**Q81. What would you ask the client?**
What each category means and the cost of each type of mistake (to set thresholds). Whether they can provide more rare-class examples, especially Category_5. What the columns really are. How often new vendors appear. Whether the one conflicting pair and the duplicates are known issues.

**Q82. What are the model's weaknesses?**
1. Rare classes have very little data; Category_5 can't be evaluated.
2. It performs best on vendors it has already seen.
3. It over-flags Category_2 (precision 76%).
4. Confidence scores aren't calibrated yet.
5. Small tuning grid, and no nested CV yet.
Being upfront about these is part of the answer.

**Q83. If the model had to be 99% accurate, what would you do?**
Not force the model; change the process. Auto-accept only high-confidence predictions (above 90–95%) and send the rest to review. Accuracy on the auto-accepted part can reach 99% or more, with a known share going to humans. Then use the reviewers' corrections as new training data.

**Q84. How would you retrain as new data arrives?**
Rerun `train.py` on the combined data (it's fully scripted and reproducible), compare the new model against the current one on a fixed benchmark set, and promote it only if macro-F1 doesn't drop.

---

## 10. Coding / technical probing

**Q85. What is a FunctionTransformer and why use it?**
It wraps a plain Python function as an sklearn step. `engineer_features` is stateless (no fitting), so it can live inside the Pipeline and gets saved together with the model.

**Q86. What is a ColumnTransformer?**
It applies different transformers to different columns (TF-IDF on each text column, one-hot on Col7, impute and scale on numeric features) and joins the results side by side into one matrix.

**Q87. Why a sparse matrix?**
6,217 features, mostly zeros. Sparse storage keeps memory small and logistic regression is fast on it. HistGradientBoosting needs dense input, which is why its vocabulary was capped and a densify step added.

**Q88. What is `sublinear_tf`?**
It uses 1 + log(term count) instead of the raw count, so a token repeated 10 times doesn't count 10 times as much.

**Q89. Why `random_state=42` everywhere?**
Reproducibility. The same split, folds and results every run, so the numbers in the dashboard, the README and this presentation all match.

**Q90. What is `cross_validate` vs `GridSearchCV`?**
`cross_validate` scores a fixed model over the folds (used to compare model families). `GridSearchCV` does that for each parameter combination and then refits the best one on all training data (used to tune the winner, with `refit="f1_macro"`).

**Q91. What happens if a new class appears in future data?**
The model can only predict classes it was trained on, so it would assign the closest known class, probably with low confidence. Monitoring the low-confidence share and reviewer corrections would catch it, and then we retrain with the new class.

---

## 11. Demo-day checklist

- [ ] `streamlit run app.py` starts cleanly; do a full run-through the day before
- [ ] OpenAI key in `.streamlit/secrets.toml` (not typed in live)
- [ ] Test the offline fallback once with Wi-Fi off
- [ ] Have `holdout_test.csv` and `holdout_test_with_labels.csv` in an easy-to-find folder
- [ ] A "messy" test file ready (renamed headers, a missing column) to show robustness
- [ ] Rows picked in advance to Explain: 1 confident C1, 1 **misclassified** row, 1 rare-class row
- [ ] Laptop charger, HDMI adapter, notifications off, browser zoom ~125% for the big screen

## 12. Questions to ask the panel at the end
- How does your team currently classify these transactions, and where does it hurt most?
- How would a model like this be handed over to a client: your platform or theirs?
- What does success look like in the first 6 months for this role?

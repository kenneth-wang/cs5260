# Data

For this project, the QA corpus was scrapped from [this](https://ask.gov.sg/agency/moh?topics=COVID-19%20Vaccination%20and%20Booster) website.

Once the QA pairs are download, put them into a Comma-Seperated (CSV) file with the following columns.

| Column name | Type | Description | Example |
| --- | --- | --- | --- |
| idx_of_ans | int | Unique identifier for each answer | 1, 2, 101 etc.|
| ans_str | str | Answer text | |
| query_str | str | Question text | |
| idx_of_qns | int | Unique identifier for each answer |  1, 2, 101 etc. |

After questions have been generated using the generative model, the new questions will need to be
saved in the same csv format. Take care to use a unique question index (idx_of_qns) for each new question.


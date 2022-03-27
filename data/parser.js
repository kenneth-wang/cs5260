const fs = require("fs");

const convertToCsv = (arr) => {
  const array = [Object.keys(arr[0])].concat(arr);
  return array.map((it) => Object.values(it).toString()).join("\n");
};

const stripHtml = (html) => {
  return html.replace(/<[^>]*>?/gm, "");
};

const csvCompatible = (str) => {
  return '"' + str.replace(/\n/g, " ").replace(/"/g, "'").trim() + '"';
};

const filenames = [
  "topic1.json",
  "topic2.json",
  "topic3.json",
  "topic4.json",
  "topic5.json",
  "topic6.json",
  "topic7.json",
  "topic8.json",
  "topic11.json",
  "topic21.json",
];

const questionAnswerPairs = filenames
  .map((filename) => fs.readFileSync(filename))
  .flatMap((data) => {
    const json = JSON.parse(data);
    return json.posts;
  })
  .map((post) => ({
    ans_str: csvCompatible(stripHtml(post.answers[0].body)),
    query_str: csvCompatible(post.title),
    idx_of_ans: post.answers[0].id,
    idx_of_qns: post.id,
  }));

const csvString = convertToCsv(questionAnswerPairs);

fs.writeFile("data.csv", csvString, (err) => {
  if (err) return console.log(err);
});

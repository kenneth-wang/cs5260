const fs = require('fs');

const convertToCSV = (arr) => {
    const array = [Object.keys(arr[0])].concat(arr);
    return array.map(it =>
        Object.values(it).toString()
    ).join('\n\n');
}

const strip = (html) => {
    return html.replace(/<[^>]*>?/gm, '');
 }

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

const pairs = filenames
    .map(filename =>
        fs.readFileSync(filename)
    )
    .flatMap(data => {
        const json = JSON.parse(data);
        return json.posts;
    })
    .map(post => ({
        ansStr: strip(post.answers[0].body),
        queryStr: post.title,
        idxOfAns: post.answers[0].id,
        idxOfQns: post.id,
    }));

console.log(convertToCSV(pairs));

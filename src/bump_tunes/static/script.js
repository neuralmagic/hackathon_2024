import abcjs from "abcjs";
abcjs.renderAbc("paper", "X:1\nK:D\nDD AA|BBA2|\n");
fetch("http://localhost:8000/api")
  .then((res) => res.json())
  .then((res) => console.log(res));

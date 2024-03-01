import abcjs from "abcjs";

const defaultABC =
  'X:1\nL:1/8\nM:2/4\nK:Dmin\n|:"Dm" DF ED | G F3 | DF ED | G F3 |"Gm" F{A}G GG | G2 G2 |1"Dm" F/E/D"Gm" GD |"Dm" F3"A7" A, :|2 \n';

// abcjs.renderAbc("paper", defaultABC);

const promptInput = document.getElementById("prompt-input");
const form = document.getElementById("form");

const handleReponse = function (music_segment) {
  setTune(false, music_segment);
  // abcjs.renderAbc("paper", music_segment);
  // promptInput.value = "";
  // const mdlTextfield = document.querySelector(".mdl-textfield");
  // if (mdlTextfield) {
  //   mdlTextfield.MaterialTextfield.change("");
  // }
};

const handleError = function (invalid_text) {
  document.getElementById("error").innerText = "Error: Try another prompt";
  document.getElementById("invalid-response").innerText = invalid_text;
  setTimeout(() => {
    document.getElementById("error").innerText = "";
    document.getElementById("invalid-response").innerText = "";
  }, 4000);
};

const makePromptRequest = function (prompt) {
  const data = {
    prompt,
    temperature: null,
    max_tokens: null,
    top_k: null,
    top_p: null,
    n: null,
  };
  fetch("http://localhost:8080/api", {
    method: "POST",
    mode: "cors",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((res) => res.json())
    .then((res) => {
      console.log(res);
      if (res.is_valid) {
        handleReponse(res.music_segment);
      } else {
        handleError(res.music_segment);
      }
    });
};

const submitPrompt = function (e) {
  e.preventDefault();
  const value = promptInput.value;
  makePromptRequest(value);
};

promptInput.onsubmit = submitPrompt;
form.onsubmit = submitPrompt;

var abcOptions = {
  add_classes: true,
  responsive: "resize",
};

function CursorControl() {
  var self = this;

  self.onReady = function () {};
  self.onStart = function () {
    var svg = document.querySelector("#paper svg");
    var cursor = document.createElementNS("http://www.w3.org/2000/svg", "line");
    cursor.setAttribute("class", "abcjs-cursor");
    cursor.setAttributeNS(null, "x1", 0);
    cursor.setAttributeNS(null, "y1", 0);
    cursor.setAttributeNS(null, "x2", 0);
    cursor.setAttributeNS(null, "y2", 0);
    svg.appendChild(cursor);
  };
  self.beatSubdivisions = 2;
  self.onBeat = function (beatNumber, totalBeats, totalTime) {};
  self.onEvent = function (ev) {
    if (ev.measureStart && ev.left === null) return; // this was the second part of a tie across a measure line. Just ignore it.

    var lastSelection = document.querySelectorAll("#paper svg .highlight");
    for (var k = 0; k < lastSelection.length; k++)
      lastSelection[k].classList.remove("highlight");

    for (var i = 0; i < ev.elements.length; i++) {
      var note = ev.elements[i];
      for (var j = 0; j < note.length; j++) {
        note[j].classList.add("highlight");
      }
    }

    var cursor = document.querySelector("#paper svg .abcjs-cursor");
    if (cursor) {
      cursor.setAttribute("x1", ev.left - 2);
      cursor.setAttribute("x2", ev.left - 2);
      cursor.setAttribute("y1", ev.top);
      cursor.setAttribute("y2", ev.top + ev.height);
    }
  };
  self.onFinished = function () {
    var els = document.querySelectorAll("svg .highlight");
    for (var i = 0; i < els.length; i++) {
      els[i].classList.remove("highlight");
    }
    var cursor = document.querySelector("#paper svg .abcjs-cursor");
    if (cursor) {
      cursor.setAttribute("x1", 0);
      cursor.setAttribute("x2", 0);
      cursor.setAttribute("y1", 0);
      cursor.setAttribute("y2", 0);
    }
  };
}

var cursorControl = new CursorControl();

var synthControl;

function load() {
  if (abcjs.synth.supportsAudio()) {
    synthControl = new abcjs.synth.SynthController();
    synthControl.load("#audio", cursorControl, {
      displayLoop: true,
      displayRestart: true,
      displayPlay: true,
      displayProgress: true,
      displayWarp: true,
    });
  } else {
    document.querySelector("#audio").innerHTML =
      "<div class='audio-error'>Audio is not supported in this browser.</div>";
  }
  setTune(false, defaultABC);
}

function setTune(userAction, musicSegment) {
  if (abcjs.synth.supportsAudio()) {
    synthControl = new abcjs.synth.SynthController();
    synthControl.load("#audio", cursorControl, {
      displayLoop: true,
      displayRestart: true,
      displayPlay: true,
      displayProgress: true,
      displayWarp: true,
    });
  } else {
    document.querySelector("#audio").innerHTML =
      "<div class='audio-error'>Audio is not supported in this browser.</div>";
  }
  synthControl.disable(true);
  var visualObj = abcjs.renderAbc("paper", musicSegment, abcOptions)[0];
  console.log("🚀 ~ setTune ~ musicSegment:", musicSegment);

  var midiBuffer = new abcjs.synth.CreateSynth();
  midiBuffer
    .init({
      visualObj: visualObj,
    })
    .then(function (response) {
      console.log(response);
      if (synthControl) {
        synthControl
          .setTune(visualObj, userAction)
          .then(function (response) {
            console.log("🚀 ~ response:", response);
            console.log("Audio successfully loaded.");
          })
          .catch(function (error) {
            console.warn("Audio problem:", error);
          });
      }
    })
    .catch(function (error) {
      console.warn("Audio problem:", error);
    });
}

document.body.onload = load;

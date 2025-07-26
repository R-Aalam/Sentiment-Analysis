// // // // // let pieChart;

// // // // // function updateSentimentChart(positive, neutral, negative) {
// // // // //   const ctx = document.getElementById("sentimentPieChart").getContext("2d");

// // // // //   if (pieChart) pieChart.destroy();

// // // // //   pieChart = new Chart(ctx, {
// // // // //     type: "pie",
// // // // //     data: {
// // // // //       labels: ["Positive", "Neutral", "Negative"],
// // // // //       datasets: [{
// // // // //         data: [positive, neutral, negative],
// // // // //         backgroundColor: ["#4CAF50", "#FFEB3B", "#F44336"],
// // // // //         borderColor: "#ffffff",
// // // // //         borderWidth: 2
// // // // //       }]
// // // // //     },
// // // // //     options: {
// // // // //       responsive: true,
// // // // //       plugins: {
// // // // //         legend: { position: "bottom" },
// // // // //         title: {
// // // // //           display: true,
// // // // //           text: "Sentiment Distribution"
// // // // //         }
// // // // //       }
// // // // //     }
// // // // //   });
// // // // // }

// // // // // async function analyzeText() {
// // // // //   const inputText = document.getElementById("textInput").value.trim();
// // // // //   if (!inputText) return alert("Please enter some text.");

// // // // //   const res = await fetch("/analyze", {
// // // // //     method: "POST",
// // // // //     headers: { "Content-Type": "application/json" },
// // // // //     body: JSON.stringify({ text: inputText })
// // // // //   });

// // // // //   const result = await res.json();
// // // // //   processSentimentData(result);
// // // // // }

// // // // // async function analyzeFile() {
// // // // //   const file = document.getElementById("fileInput").files[0];
// // // // //   if (!file) return alert("Please select a file.");

// // // // //   const formData = new FormData();
// // // // //   formData.append("file", file);

// // // // //   const res = await fetch("/analyze-file", {
// // // // //     method: "POST",
// // // // //     body: formData
// // // // //   });

// // // // //   const result = await res.json();
// // // // //   processSentimentData(result)
// // // // // }

// // // // // function processSentimentData(data) {
// // // // //   const counts = { Positive: 0, Neutral: 0, Negative: 0 };

// // // // //   data.sentiment_class.forEach(label => {
// // // // //     if (label === "Positive" || label === 2) counts.Positive++;
// // // // //     else if (label === "Neutral" || label === 1) counts.Neutral++;
// // // // //     else if (label === "Negative" || label === 0) counts.Negative++;
// // // // //   });

// // // // //   updateSentimentChart(counts.Positive, counts.Neutral, counts.Negative);

// // // // // }
// // // // // function filterSentiments(type) {
// // // // //   console.log("Filtering for:", type);
// // // // //   console.log("All results:", allResults);

// // // // //   const filtered = allResults.filter(item => item.sentiment === type);
// // // // //   const container = document.getElementById("filteredResults");

// // // // //   if (filtered.length === 0) {
// // // // //     container.innerHTML = `<p>No ${type.toLowerCase()} comments found.</p>`;
// // // // //     return;
// // // // //   }

// // // // //   container.innerHTML = `
// // // // //     <h4>${type} Comments:</h4>
// // // // //     <ul>${filtered.map(item => `<li>${item.text}</li>`).join("")}</ul>
// // // // //   `;
// // // // // }
// // // let pieChart;
// // // let allResults = [];  // ✅ Store results globally

// // // function updateSentimentChart(positive, neutral, negative) {
// // //   const ctx = document.getElementById("sentimentPieChart").getContext("2d");

// // //   if (pieChart) pieChart.destroy();

// // //   pieChart = new Chart(ctx, {
// // //     type: "pie",
// // //     data: {
// // //       labels: ["Positive", "Neutral", "Negative"],
// // //       datasets: [{
// // //         data: [positive, neutral, negative],
// // //         backgroundColor: ["#4CAF50", "#FFEB3B", "#F44336"],
// // //         borderColor: "#ffffff",
// // //         borderWidth: 2
// // //       }]
// // //     },
// // //     options: {
// // //       responsive: true,
// // //       plugins: {
// // //         legend: { position: "bottom" },
// // //         title: {
// // //           display: true,
// // //           text: "Sentiment Distribution"
// // //         }
// // //       }
// // //     }
// // //   });
// // // }

// // // async function analyzeText() {
// // //   const inputText = document.getElementById("textInput").value.trim();
// // //   if (!inputText) return alert("Please enter some text.");

// // //   const res = await fetch("/analyze", {
// // //     method: "POST",
// // //     headers: { "Content-Type": "application/json" },
// // //     body: JSON.stringify({ text: inputText })
// // //   });

// // //   const result = await res.json();
// // //   processSentimentData(result);
// // // }

// // // async function analyzeFile() {
// // //   const file = document.getElementById("fileInput").files[0];
// // //   if (!file) return alert("Please select a file.");

// // //   const formData = new FormData();
// // //   formData.append("file", file);

// // //   const res = await fetch("/analyze-file", {
// // //     method: "POST",
// // //     body: formData
// // //   });

// // //   const result = await res.json();
// // //   processSentimentData(result);
// // // }

// // // function processSentimentData(data) {
// // //   const counts = { Positive: 0, Neutral: 0, Negative: 0 };
// // //   allResults = [];  // ✅ Reset

// // //   const sentiments = data.sentiment_class;
// // //   const labels = Array.isArray(data.label) ? data.label : [data.label];

// // //   for (let i = 0; i < sentiments.length; i++) {
// // //     let sentiment = "";
// // //     const label = sentiments[i];

// // //     if (label === 2 || label === "Positive") {
// // //       sentiment = "Positive";
// // //       counts.Positive++;
// // //     } else if (label === 1 || label === "Neutral") {
// // //       sentiment = "Neutral";
// // //       counts.Neutral++;
// // //     } else {
// // //       sentiment = "Negative";
// // //       counts.Negative++;
// // //     }

// // //     allResults.push({
// // //       sentiment: sentiment,
// // //       text: labels[i] || "No comment"
// // //     });
// // //   }

// // //   updateSentimentChart(counts.Positive, counts.Neutral, counts.Negative);
// // //   document.getElementById("filteredResults").innerHTML = ""; // Clear previous
// // // }

// // // function filterSentiments(type) {
// // //   console.log("Filtering for:", type);
// // //   console.log("All results:", allResults);

// // //   const filtered = allResults.filter(item => item.sentiment === type);
// // //   const container = document.getElementById("filteredResults");

// // //   if (filtered.length === 0) {
// // //     container.innerHTML = `<p>No ${type.toLowerCase()} comments found.</p>`;
// // //     return;
// // //   }

// // //   container.innerHTML = `
// // //     <h4>${type} Comments:</h4>
// // //     <ul>${filtered.map(item => `<li>${item.text}</li>`).join("")}</ul>
// // //   `;
// // // }
// // window.addEventListener("DOMContentLoaded", () => {
// //   document.getElementById("filterButtons").style.display = "none";
// // });

// // let pieChart;
// // let allResults = [];  // ✅ Store results globally

// // function updateSentimentChart(positive, neutral, negative) {
// //   const ctx = document.getElementById("sentimentPieChart").getContext("2d");

// //   if (pieChart) pieChart.destroy();

// //   pieChart = new Chart(ctx, {
// //     type: "pie",
// //     data: {
// //       labels: ["Positive", "Neutral", "Negative"],
// //       datasets: [{
// //         data: [positive, neutral, negative],
// //         backgroundColor: ["#4CAF50", "#FFEB3B", "#F44336"],
// //         borderColor: "#ffffff",
// //         borderWidth: 2
// //       }]
// //     },
// //     options: {
// //       responsive: true,
// //       plugins: {
// //         legend: { position: "bottom" },
// //         title: {
// //           display: true,
// //           text: "Sentiment Distribution"
// //         }
// //       }
// //     }
// //   });
// // }

// // async function analyzeText() {
// //   const inputText = document.getElementById("textInput").value.trim();
// //   if (!inputText) return alert("Please enter some text.");

// //   const res = await fetch("/analyze", {
// //     method: "POST",
// //     headers: { "Content-Type": "application/json" },
// //     body: JSON.stringify({ text: inputText })
// //   });

// //   const result = await res.json();
// //   processSentimentData(result);

// //   // Hide buttons when analyzing text
// //   document.getElementById("filterButtons").style.display = "none";
// // }

// // async function analyzeFile() {
// //   const file = document.getElementById("fileInput").files[0];
// //   if (!file) return alert("Please select a file.");

// //   const formData = new FormData();
// //   formData.append("file", file);

// //   const res = await fetch("/analyze-file", {
// //     method: "POST",
// //     body: formData
// //   });

// //   const result = await res.json();
// //   processSentimentData(result);

// //   // Show buttons only after file upload analysis
// //   document.getElementById("filterButtons").style.display = "block";
// // }

// // function processSentimentData(data) {
// //   const counts = { Positive: 0, Neutral: 0, Negative: 0 };
// //   allResults = [];  // ✅ Reset

// //   const sentiments = data.sentiment_class;
// //   const labels = Array.isArray(data.label) ? data.label : [data.label];

// //   for (let i = 0; i < sentiments.length; i++) {
// //     let sentiment = "";
// //     const label = sentiments[i];

// //     if (label === 2 || label === "Positive") {
// //       sentiment = "Positive";
// //       counts.Positive++;
// //     } else if (label === 1 || label === "Neutral") {
// //       sentiment = "Neutral";
// //       counts.Neutral++;
// //     } else {
// //       sentiment = "Negative";
// //       counts.Negative++;
// //     }

// //     allResults.push({
// //       sentiment: sentiment,
// //       text: labels[i] || "No comment"
// //     });
// //   }

// //   updateSentimentChart(counts.Positive, counts.Neutral, counts.Negative);
// //   document.getElementById("filteredResults").innerHTML = ""; // Clear previous
// // }

// // function filterSentiments(type) {
// //   console.log("Filtering for:", type);
// //   console.log("All results:", allResults);

// //   const filtered = allResults.filter(item => item.sentiment === type);
// //   const container = document.getElementById("filteredResults");

// //   if (filtered.length === 0) {
// //     container.innerHTML = `<p>No ${type.toLowerCase()} comments found.</p>`;
// //     return;
// //   }

// //   container.innerHTML = `
// //     <h4>${type} Comments:</h4>
// //     <ul>${filtered.map(item => `<li>${item.text}</li>`).join("")}</ul>
// //   `;
// // }

// let pieChart;
// let allResults = [];  // Store results globally

// function updateSentimentChart(positive, neutral, negative) {
//   const ctx = document.getElementById("sentimentPieChart").getContext("2d");

//   if (pieChart) pieChart.destroy();

//   pieChart = new Chart(ctx, {
//     type: "pie",
//     data: {
//       labels: ["Positive", "Neutral", "Negative"],
//       datasets: [{
//         data: [positive, neutral, negative],
//         backgroundColor: ["#4CAF50", "#FFEB3B", "#F44336"],
//         borderColor: "#ffffff",
//         borderWidth: 2
//       }]
//     },
//     options: {
//       responsive: true,
//       plugins: {
//         legend: { position: "bottom" },
//         title: {
//           display: true,
//           text: "Sentiment Distribution"
//         }
//       }
//     }
//   });
// }

// async function analyzeText() {
//   const inputText = document.getElementById("textInput").value.trim();
//   if (!inputText) return alert("Please enter some text.");

//   const res = await fetch("/analyze", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ text: inputText })
//   });

//   const result = await res.json();
//   processSentimentData(result);

//   // Hide filter buttons when analyzing text input
//   document.getElementById("filterButtons").style.display = "none";
// }

// async function analyzeFile() {
//   const file = document.getElementById("fileInput").files[0];
//   if (!file) return alert("Please select a file.");

//   const formData = new FormData();
//   formData.append("file", file);

//   const res = await fetch("/analyze-file", {
//     method: "POST",
//     body: formData
//   });

//   const result = await res.json();
//   processSentimentData(result);

//   // Show filter buttons only after file upload
//   document.getElementById("filterButtons").style.display = "block";
// }

// function processSentimentData(data) {
//   const counts = { Positive: 0, Neutral: 0, Negative: 0 };
//   allResults = [];  // Reset previous data

//   const sentiments = data.sentiment_class;
//   const labels = Array.isArray(data.label) ? data.label : [data.label];
//   const scores = Array.isArray(data.sentiment_score) ? data.sentiment_score : [data.sentiment_score];

//   for (let i = 0; i < sentiments.length; i++) {
//     let sentiment = "";
//     const label = sentiments[i];

//     if (label === 2 || label === "Positive") {
//       sentiment = "Positive";
//       counts.Positive++;
//     } else if (label === 1 || label === "Neutral") {
//       sentiment = "Neutral";
//       counts.Neutral++;
//     } else {
//       sentiment = "Negative";
//       counts.Negative++;
//     }

//     allResults.push({
//       sentiment: sentiment,
//       text: labels[i] || "No comment",
//       score: scores[i] !== undefined ? scores[i].toFixed(2) : "N/A"
//     });
//   }

//   updateSentimentChart(counts.Positive, counts.Neutral, counts.Negative);
//   document.getElementById("filteredResults").innerHTML = ""; // Clear previous filtered comments
// }

// function filterSentiments(type) {
//   const filtered = allResults.filter(item => item.sentiment === type);
//   const container = document.getElementById("filteredResults");

//   if (filtered.length === 0) {
//     container.innerHTML = `<p>No ${type.toLowerCase()} comments found.</p>`;
//     return;
//   }

//   container.innerHTML = `
//     <h4>${type} Comments:</h4>
//     <ul>
//       ${filtered.map(item => `<li>${item.text} <strong>(Score: ${item.score})</strong></li>`).join("")}
//     </ul>
//   `;
// }

// // On page load, hide the filter buttons container
// window.addEventListener("DOMContentLoaded", () => {
//   document.getElementById("filterButtons").style.display = "none";
// });
let pieChart;
let allResults = [];  // Store results globally

function updateSentimentChart(positive, neutral, negative) {
  const ctx = document.getElementById("sentimentPieChart").getContext("2d");

  if (pieChart) pieChart.destroy();

  pieChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [{
        data: [positive, neutral, negative],
        backgroundColor: ["#4CAF50", "#FFEB3B", "#F44336"],
        borderColor: "#ffffff",
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        title: {
          display: true,
          text: "Sentiment Distribution"
        }
      }
    }
  });
}

async function analyzeText() {
  const inputText = document.getElementById("textInput").value.trim();
  if (!inputText) return alert("Please enter some text.");

  const res = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: inputText })
  });

  const result = await res.json();
  processSentimentData(result);

  // Hide filter buttons when analyzing text input
  document.getElementById("filterButtons").style.display = "none";
}

async function analyzeFile() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Please select a file.");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/analyze-file", {
    method: "POST",
    body: formData
  });

  const result = await res.json();
  processSentimentData(result);

  // Show filter buttons only after file upload
  document.getElementById("filterButtons").style.display = "block";
}

function processSentimentData(data) {
  const counts = { Positive: 0, Neutral: 0, Negative: 0 };
  allResults = [];  // Reset previous data

  const sentiments = data.sentiment_class;
  const labels = Array.isArray(data.label) ? data.label : [data.label];
  const scores = Array.isArray(data.sentiment_score) ? data.sentiment_score : [data.sentiment_score];

  for (let i = 0; i < sentiments.length; i++) {
    let sentiment = "";
    const label = sentiments[i];

    if (label === 2 || label === "Positive") {
      sentiment = "Positive";
      counts.Positive++;
    } else if (label === 1 || label === "Neutral") {
      sentiment = "Neutral";
      counts.Neutral++;
    } else {
      sentiment = "Negative";
      counts.Negative++;
    }

    allResults.push({
      sentiment: sentiment,
      text: labels[i] || "No comment",
      score: scores[i] !== undefined ? scores[i] : 0
    });
  }

  updateSentimentChart(counts.Positive, counts.Neutral, counts.Negative);
  document.getElementById("filteredResults").innerHTML = ""; // Clear previous filtered comments
}

function filterSentiments(type) {
  const filtered = allResults.filter(item => item.sentiment === type);
  const container = document.getElementById("filteredResults");

  if (filtered.length === 0) {
    container.innerHTML = `<p>No ${type.toLowerCase()} comments found.</p>`;
    return;
  }

  container.innerHTML = `
    <h4>${type} Comments:</h4>
    <ul class="comment-list">
      ${filtered.map(item => `
        <li class="comment-item">
          <span class="comment-text">${item.text}</span>
          <span class="comment-score">${Math.round(item.score * 100)}%</span>
        </li>`).join("")}
    </ul>
  `;
}

// On page load, hide the filter buttons container
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("filterButtons").style.display = "none";
});

// hit “Save score” button after training
async function saveScore(name, valLoss) {
  await fetch('/api/server?what=leaderboard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, valLoss })
  });
}

// on page load get top 50
(async () => {
  const board = await fetch('/api/server?what=leaderboard').then(r => r.json());
  console.table(board);
})();

// Get weather data
(async () => {
  const weatherData = await fetch('/api/server?what=weather').then(r => r.json());
  console.log("Weather data from server:", weatherData);
  document.getElementById('texttt').textContent = `Current temperature in NYC: ${weatherData.current_weather.temperature}°C`;
})();

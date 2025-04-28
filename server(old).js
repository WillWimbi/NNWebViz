// server.js  (root or /server folder – either way works)

require('dotenv').config();                       // ← load .env if you add one
const express  = require('express');
const mongoose = require('mongoose');
const cors     = require('cors');
const path     = require('path');



const MONGO_URI = process.env.MONGO_URI ||
  'mongodb+srv://willwimbiscus:9Aug2005%21@cluster0.n0ocjfm.mongodb.net/mnistLab?retryWrites=true&w=majority';
const PORT = process.env.PORT || 4000;

const app = express();
app.use(cors());
app.use(express.json());

// ----------- serve static UI (index.html, script.js…) -------------
app.use(express.static(path.join(__dirname, 'public')));   // / → index.html

// ----------- Mongo connection -------------
mongoose.connect(MONGO_URI)
        .then(() => console.log('✅ Mongo connected'))
        .catch(err  => console.error('Mongo error', err));

const leaderSchema = new mongoose.Schema({
  name:     { type: String, required: true },
  valLoss:  { type: Number, required: true },
  createdAt:{ type: Date, default: Date.now }
});
const Leader = mongoose.model('Leader', leaderSchema);

// ----------- REST API ---------------------
app.get('/api/leaderboard', async (req, res) => {
  const top = await Leader.find().sort({ valLoss: 1 }).limit(50);
  res.json(top);
});

app.post('/api/leaderboard', async (req, res) => {
  const { name, valLoss } = req.body;
  if (!name || typeof valLoss !== 'number') {
    return res.status(400).send('Bad data');
  }
  await Leader.create({ name, valLoss });
  res.json({ ok: true });
});


// ----------- start server -----------------
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));

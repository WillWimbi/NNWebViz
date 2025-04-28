import { MongoClient, ServerApiVersion } from 'mongodb';

const uri = process.env.MONGO_URI;     // set in Vercel dashboard

/* ---------- reuse client across cold starts ---------- */
let cached = global._mongoClient;
if (!cached) {
  cached = global._mongoClient = new MongoClient(uri, {
    serverApi: ServerApiVersion.v1
  });
}
const db = cached.db('mnistLab');

export default async function handler(req, res) {
  const { method, query } = req;
  const what = query.what;                   // ?what=leaderboard | nets

  /* ---------- LEADERBOARD ---------- */
  if (method === 'GET' && what === 'leaderboard') {
    const list = await db.collection('leaderboard')
                         .find().sort({ valLoss: 1 }).limit(50).toArray();
                         // .sort({ valLoss: 1 }) sorts the results in ascending order by valLoss
                         // This means the lowest validation loss scores will appear first
                         // Using -1 instead would sort in descending order
    return res.status(200).json(list);
  }

  if (method === 'POST' && what === 'leaderboard') {
    const { name, valLoss } = req.body || {};
    if (!name || typeof valLoss !== 'number')
      return res.status(400).json({ error: 'Bad payload' });

    await db.collection('leaderboard').insertOne({ name, valLoss, ts: new Date() });
    return res.json({ ok: true });
  }

  /* ---------- 100 PRETRAINED NETS ---------- */
  if (method === 'GET' && what === 'nets') {
    const nets = await db.collection('pretrained100').find().toArray();
    return res.json(nets);
  }

  /* ---------- WEATHER DATA ---------- */
  if (method === 'GET' && what === 'weather') {
    const city = query.city || 'New York';
    // NYC coordinates: 40.7128° N, 74.0060° W
    const url = `https://api.open-meteo.com/v1/forecast?current_weather=true&timezone=UTC&latitude=40.7128&longitude=-74.0060&city=${encodeURIComponent(city)}`;
    const weatherData = await fetch(url).then(r => r.json());
    return res.status(200).json(weatherData);
  }

  /* ---------- fall-through ---------- */
  res.setHeader('Allow', ['GET', 'POST']);
  res.status(405).end();
}

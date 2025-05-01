// api/runs.js
import { MongoClient } from 'mongodb';

let cached = null;

export default async function handler(req, res) {
  try {
    if (!cached) {
      cached = new MongoClient(process.env.MONGODB_URI);
      await cached.connect();
    }
    const runs = await cached
      .db('mnistLab')                   // your db
      .collection('staticPretrained')   // your collection
      .find({ runId: /^idx/ })
      .sort({ runId: 1 })               // idx00 → idx49
      .toArray();

    res.setHeader('Cache-Control', 'no-store');
    return res.status(200).json(runs);
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'db-error' });
  }
}

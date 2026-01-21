const admin = require("firebase-admin");
const serviceAccount = require("./serviceAccountKey.json");

// Init Firebase
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

// Firestore reference
const db = admin.firestore();

module.exports = { admin, db };

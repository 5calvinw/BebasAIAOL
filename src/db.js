import Dexie from "dexie";

export const db = new Dexie("AOL_DB");

db.version(1).stores({
  collection: "collectionID, plasticTypeID, condition, confidence, status",
  plastictype: "plasticTypeID, timestamp, RICNumber, difficulty, foundin, description"
});

export async function seedPlasticTypes() {
  const count = await db.plastictype.count();
  if (count === 0) {
    await db.plastictype.bulkAdd([
  {
    plasticTypeID: 1,
    RICNumber: 1,
    difficulty: "easy",
    foundin: "Water bottles, soda bottles, food packaging",
    description:
      "PET (Polyethylene Terephthalate) is widely used for food and drink packaging because it prevents oxygen from spoiling products and keeps carbonation inside drinks. It is also known as a wrinkle-free fiber and contains antimony trioxide, a carcinogen, but remains highly recyclable. Recycled PET becomes plastic flakes or pellets used to produce new containers."
  },
  {
    plasticTypeID: 2,
    RICNumber: 2,
    difficulty: "easy",
    foundin: "Milk jugs, detergent bottles, shampoo bottles, motor oil bottles",
    description:
      "HDPE (High-Density Polyethylene) is known for its tensile strength, high impact resistance, and melting point. Industrial-grade and food-safe, it is widely used for beverage and household containers. It is more stable and thicker than PET, easily recyclable, and becomes flakes or pellets used to make pipes, crates, plastic lumber, and recycling bins."
  },
  {
    plasticTypeID: 3,
    RICNumber: 3,
    difficulty: "difficult",
    foundin: "Pipes, toys, cling wrap, medical tubing",
    description:
      "PVC (Polyvinyl Chloride) is durable, strong, light, and fire-resistant because of added plasticisers and stabilisers. It is common in construction materials and medical supplies. PVC is rarely recycled due to environmental risk: melting releases harmful chemicals. It can be recycled by mechanical or chemical methods but requires specialised facilities."
  },
  {
    plasticTypeID: 4,
    RICNumber: 4,
    difficulty: "difficult",
    foundin: "Plastic bags, plastic wrap, food storage containers, infusion bottles",
    description:
      "LDPE (Low-Density Polyethylene) is flexible, durable, and moisture-resistant. It is widely used for grocery bags, plastic wraps, squeezable bottles, and container lids. LDPE is hard to recycle due to low density and high contamination risk. Where facilities exist, recycled LDPE becomes plastic pellets used for lumber, outdoor furniture, and drainage pipes."
  },
  {
    plasticTypeID: 5,
    RICNumber: 5,
    difficulty: "medium",
    foundin: "Fruit containers, plastic furniture, microwaveable meal trays, vehicle parts",
    description:
      "PP (Polypropylene) is resistant to heat and chemicals, used for hot-food containers and automotive parts. It is partially recyclable and may cause hormone disruption and asthma when improperly handled. Recycled PP is often used for industrial products like battery boxes and signal lights, but it can only be recycled up to four times before quality degrades."
  },
  {
    plasticTypeID: 6,
    RICNumber: 6,
    difficulty: "difficult",
    foundin: "Disposable cups, plastic cutlery, egg cartons",
    description:
      "PS (Polystyrene), also known as Styrofoam, can release harmful styrene chemicals when heated and is rarely recycled due to high processing costs and environmental risk."
  },
  {
    plasticTypeID: 7,
    RICNumber: 7,
    difficulty: "difficult",
    foundin: "Baby bottles, sippy cups, water bottles, medical storage containers",
    description:
      "‘Other plastics’ include polycarbonate and bioplastics. They are made from mixed or layered materials, sometimes containing BPA, making them difficult to recycle without specialised technology and equipment."
  }
]);

    console.log("plastictype table populated ✓");
  }
}

export async function seedCollections() {
  const count = await db.collection.count();
  if (count === 0) {
    const now = new Date();
    await db.collection.bulkAdd([
      { collectionID: 1, plasticTypeID: 1, condition: "Clean", confidence: 0.95, status: "New", timestamp: now },
      { collectionID: 2, plasticTypeID: 2, condition: "Dirty", confidence: 0.85, status: "In Progress", timestamp: new Date(now.getTime() - 1000 * 60 * 60) },
      { collectionID: 3, plasticTypeID: 3, condition: "Clean", confidence: 0.75, status: "New", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 2) },
      { collectionID: 4, plasticTypeID: 4, condition: "Dirty", confidence: 0.65, status: "Reviewed", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 3) },
      { collectionID: 5, plasticTypeID: 5, condition: "Clean", confidence: 0.90, status: "New", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 4) },
      { collectionID: 6, plasticTypeID: 6, condition: "Dirty", confidence: 0.60, status: "In Progress", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 5) },
      { collectionID: 7, plasticTypeID: 7, condition: "Clean", confidence: 0.80, status: "Reviewed", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 6) },
      { collectionID: 8, plasticTypeID: 1, condition: "Dirty", confidence: 0.70, status: "New", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 7) },
      { collectionID: 9, plasticTypeID: 2, condition: "Clean", confidence: 0.88, status: "In Progress", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 8) },
      { collectionID: 10, plasticTypeID: 3, condition: "Dirty", confidence: 0.55, status: "Reviewed", timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 9) }
    ]);

    console.log("collection table populated ✓");
  }
}

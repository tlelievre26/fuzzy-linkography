// import embeddings lib
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0/dist/transformers.min.js";
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const DIMENSION = 384;
// import react
const e = React.createElement;

/// global graph properties

const GRAPH_WIDTH = 1000;
const INIT_X = 10;
const INIT_Y = 500;
const MIN_LINK_STRENGTH = 0.35;

/// design moves

const ideas0a = {
	title: "Fully connected",
	moves: [
		{text: "hello"},
		{text: "hello"},
		{text: "hello"},
		{text: "hello"},
		{text: "hello"},
		{text: "hello"},
		{text: "hello"},
	]
};

const ideas0b = {
	title: "No connection",
	moves: [
		{text: "wolves"},
		{text: "snuggly"},
		{text: "dishpan"},
		{text: "cranium"},
		{text: "disruptor"},
		{text: "legate"},
		{text: "orange"},
	]
}

const ideas1 = {
	title: "Stream-of-consciousness ideation (Max)",
	moves: [
		{text: "hello"},
		{text: "hello world"},
		{text: "dog"},
		{text: "cat"},
		{text: "zebra"},
		{text: "horse"},
		{text: "hello horse world"},
		{text: "hello kitty"},
		{text: "stream of consciousness"},
		{text: "kitty cat"},
		{text: "not much money in the kitty"},
		{text: "piggy bank"},
		{text: "riverbank"},
		{text: "stream"},
		{text: "video stream"},
		{text: "cat video"},
		{text: "hello kitty tv show"},
	],
};

const ideas2 = {
	title: "Stream-of-consciousness ideation (Isaac)",
	moves: [
		{text: "a phrase"},
		{text: "streetlights"},
		{text: "LED"},
		{text: "ceiling fixture"},
		{text: "buzzing"},
		{text: "Christmas lights"},
		{text: "giant skeleton"},
		{text: "trains"},
		{text: "railroad tracks"},
		{text: "Alfred Hitchcock"},
		{text: "cinematography"},
		{text: "The Fall"},
		{text: "red"},
		{text: "billowing cloth"},
		{text: "lens flare"},
		{text: "Industrial Light and Magic"},
		{text: "blue screen"},
	]
};

const ideas3 = {
	title: "Developer Diary",
	moves: [
		{text: "Photorealistic environments"},
		{text: "Fully automatic weapon-equipped vehicles"},
		{text: "Highly detailed and interactive urban environments"},
		{text: "Destructive racing"},
		{text: "40-hour-plus main quest"},
		{text: "Squads of adventurers"},
		{text: "Underwater levels"},
		{text: "Ice levels"},
		{text: "Lava levels"},
		{text: "Fire levels"},
		{text: "Fighting the occult"},
		{text: "Water effects"},
		{text: "3D layering"},
		{text: "Stone traps"},
		{text: "Wood traps"},
		{text: "Environmental hazards"},
		{text: "Vehicle sections"},
		{text: "Sniper sections"},
		{text: "Stealth sections"},
		{text: "Vehicle sections"},
		{text: "Two vehicle sections"},
		{text: "Airplanes"},
		{text: "Cars"},
		{text: "Driving"},
		{text: "Swords"},
		{text: "Fire"},
		{text: "Religion"},
		{text: "Politics"},
	]
};

/// math utils

function sum(xs) {
	return xs.reduce((a, b) => a + b, 0);
}

function dotProduct(vectorA, vectorB) {
	let dotProd = 0;
	for (let comp = 0; comp < DIMENSION; comp++) {
		dotProd += vectorA[comp] * vectorB[comp];
	}
	return dotProd;
}

function magnitude(vector) {
	return Math.sqrt(dotProduct(vector, vector));
}

function cosineSimilarity(vectorA, vectorB) {
	const dotProd = dotProduct(vectorA, vectorB);
	const magProd = magnitude(vectorA) * magnitude(vectorB);
	return dotProd / magProd;
}

// Translate `num` from the scale bounded by `oldMin` and `oldMax`
// to the scale bounded by `newMin` and `newMax`.
function scale(num, [oldMin, oldMax], [newMin, newMax]) {
	const oldRange = oldMax - oldMin;
	const newRange = newMax - newMin;
	return (((num - oldMin) / oldRange) * newRange) + newMin;
}

/// stats on a computed linkograph

function computeLinkIndexes(graph) {
	for (let i = 0; i < graph.moves.length; i++) {
		graph.moves[i].backlinkIndex = sum(
			Object.values(graph.links[i])
				.filter(n => n >= MIN_LINK_STRENGTH)
				.map(n => scale(n, [MIN_LINK_STRENGTH, 1], [0, 1]))
		);
		graph.moves[i].forelinkIndex = sum(
			Object.values(graph.links).map(linkSet => linkSet[i] || 0)
				.filter(n => n >= MIN_LINK_STRENGTH)
				.map(n => scale(n, [MIN_LINK_STRENGTH, 1], [0, 1]))
		); 
	}
}

/// data processing

async function embed(str) {
	return (await extractor(str, {convert_to_tensor: "True", pooling: "mean", normalize: true}))[0];
}

async function deriveLinks(moves) {
	// calculate embeddings for each move
	for (const move of moves) {
		move.embedding = (await embed(move.text)).tolist();
	}
	// build links table
	const links = {};
	for (let i = 0; i < moves.length; i++) {
		const currMove = moves[i];
		links[i] = {};
		for (let j = 0; j < i; j++) {
			const prevMove = moves[j];
			links[i][j] = cosineSimilarity(currMove.embedding, prevMove.embedding);
		}
	}
	return links;
}

/// react rendering

// Given the rendered locations of two design moves, return the location of the
// right-angle "elbow" between them.
function elbow(pt1, pt2) {
	const x = (pt1.x + pt2.x) / 2;
	const y = pt1.y - ((pt2.x - pt1.x) / 2);
	return {x, y};
}

// Given `props` augmented with the `idx` of a design move, return the location
// at which this move should be rendered.
function moveLoc(props) {
	return {x: (props.idx * props.moveSpacing) + INIT_X, y: INIT_Y};
}

function DesignMove(props) {
	const move = props.moves[props.idx];
	const currLoc = moveLoc(props);
	return e("g", {},
		e("text", {
			x: currLoc.x + 5, y: currLoc.y - 10,
			transform: `rotate(270, ${currLoc.x + 5}, ${currLoc.y - 10})`
		}, move.text),
		e("text", {
			x: currLoc.x + 5, y: currLoc.y + 5,
			transform: `rotate(270, ${currLoc.x + 5}, ${currLoc.y - 10})`,
			fontSize: "smaller", fill: "#aaa",
		}, `(→ ${move.forelinkIndex.toFixed(2)}, ← ${move.backlinkIndex.toFixed(2)})`),
		e("circle", {cx: currLoc.x, cy: currLoc.y, fill: "red", r: 5})
	);
}

function makeLinkObjects(props) {
	const linkLines = [];
	const linkJoints = [];
	for (const [currIdx, linkSet] of Object.entries(props.links)) {
		const currLoc = moveLoc({...props, idx: currIdx});
		for (const [prevIdx, strength] of Object.entries(linkSet)) {
			if (strength < MIN_LINK_STRENGTH) continue; // skip weak connections (arbitrary threshold)
			const prevLoc = moveLoc({...props, idx: prevIdx});
			const jointLoc = elbow(currLoc, prevLoc);
			const lineStrength = scale(strength, [MIN_LINK_STRENGTH, 1], [255, 0]);
			const color = `rgb(${lineStrength},${lineStrength},${lineStrength})`;
			linkLines.push({
				x1: currLoc.x, y1: currLoc.y, x2: jointLoc.x, y2: jointLoc.y, color, strength,
			});
			linkLines.push({
				x1: prevLoc.x, y1: prevLoc.y, x2: jointLoc.x, y2: jointLoc.y, color, strength,
			});
			linkJoints.push({x: jointLoc.x, y: jointLoc.y, color, strength});
		}
	}
	return {linkLines, linkJoints};
}

function FuzzyLinkograph(props) {
	const {linkLines, linkJoints} = makeLinkObjects(props);
	return e("div", {className: "fuzzy-linkograph"},
		e("h2", {}, props.title),
		e("svg", {viewBox: `0 0 ${GRAPH_WIDTH} ${(GRAPH_WIDTH / 2) + INIT_Y}`},
			...linkLines.sort((a, b) => a.strength - b.strength).map(line => {
				return e("line", {x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2, stroke: line.color});
			}),
			...linkJoints.sort((a, b) => a.strength - b.strength).map(joint => {
				return e("circle", {cx: joint.x, cy: joint.y, r: 5, fill: joint.color});
			}),
			...props.moves.map((_, idx) => e(DesignMove, {...props, idx}))
		)
	);
}

/// top-level app init

const appState = {
	ideaSets: [ideas0a, ideas0b, ideas1, ideas2, ideas3]
};

let root = null;
function renderUI() {
	if (!root) {
		root = ReactDOM.createRoot(document.getElementById('app'));
	}
	root.render(e("div", {},
		...appState.ideaSets.map(ideas => e(FuzzyLinkograph, ideas))
	));
}

async function main() {
	for (const ideaSet of appState.ideaSets) {
		ideaSet.links = await deriveLinks(ideaSet.moves);
		computeLinkIndexes(ideaSet);
		ideaSet.moveSpacing = (GRAPH_WIDTH - (INIT_X * 4)) / (ideaSet.moves.length - 1);
		console.log(ideaSet);
	}
	renderUI();
}

main();

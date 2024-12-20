// import embeddings lib
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0/dist/transformers.min.js";
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const DIMENSION = 384;
// import react
const e = React.createElement;

/// design moves

const ideas1 = {
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

/// math utils

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

const INIT_X = 10;
const INIT_Y = 200;
const MOVE_SPACING = (1000 - (INIT_X * 2)) / Math.max(ideas1.moves.length, ideas2.moves.length);
const MIN_LINK_STRENGTH = 0.3;

// Given a design `move` augmented with an `idx`, return the location at which
// this move should be rendered.
function moveLoc(move) {
	return {x: (move.idx * MOVE_SPACING) + INIT_X, y: INIT_Y};
}

function Node(props) {
	const currLoc = moveLoc(props);
	return e("g", {},
		e("text", {
			x: currLoc.x + 5, y: currLoc.y - 10,
			transform: `rotate(270, ${currLoc.x + 5}, ${currLoc.y - 10})`
		}, props.text),
		e("circle", {cx: currLoc.x, cy: currLoc.y, fill: "red", r: 5})
	);
}

function makeLinkObjects(allLinks) {
	const linkLines = [];
	const linkJoints = [];
	for (const [currIdx, linkSet] of Object.entries(allLinks)) {
		const currLoc = moveLoc({idx: currIdx});
		for (const [prevIdx, strength] of Object.entries(linkSet)) {
			if (strength < MIN_LINK_STRENGTH) continue; // skip weak connections (arbitrary threshold)
			const prevLoc = moveLoc({idx: prevIdx});
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
	const {linkLines, linkJoints} = makeLinkObjects(props.links);
	return e("svg", {viewBox: "0 0 1000 1000"},
		...linkLines.sort((a, b) => a.strength - b.strength).map(line => {
			return e("line", {x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2, stroke: line.color});
		}),
		...linkJoints.sort((a, b) => a.strength - b.strength).map(joint => {
			return e("circle", {cx: joint.x, cy: joint.y, r: 5, fill: joint.color});
		}),
		...props.moves.map((move, idx) => e(Node, {...move, idx}))
	);
}

let root = null;
function renderUI() {
	if (!root) {
		root = ReactDOM.createRoot(document.getElementById('app'));
	}
	root.render(e("div", {},
		e(FuzzyLinkograph, ideas1),
		e(FuzzyLinkograph, ideas2)
	));
}

async function main() {
	ideas1.links = await deriveLinks(ideas1.moves);
	console.log(ideas1);
	ideas2.links = await deriveLinks(ideas2.moves);
	console.log(ideas2);
	renderUI();
}

main();

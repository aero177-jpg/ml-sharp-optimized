import {
	DEFAULT_TRAJECTORY_PARAMS,
	type TrajectoryType,
} from "./trajectory/types";
import { GaussianViewer } from "./viewer/GaussianViewer";

console.log("[main] Script loaded");

// Get DOM elements
const containerElement = document.getElementById("canvas-container");
console.log("[main] Container element:", containerElement);
const fileLoaderElement = document.getElementById("file-loader");
const fileInputElement = document.getElementById(
	"file-input",
) as HTMLInputElement;
const trajectorySelectElement = document.getElementById(
	"trajectory-select",
) as HTMLSelectElement;
const playButtonElement = document.getElementById(
	"play-btn",
) as HTMLButtonElement;
const pauseButtonElement = document.getElementById(
	"pause-btn",
) as HTMLButtonElement;
const resetButtonElement = document.getElementById(
	"reset-btn",
) as HTMLButtonElement;
const loadingElement = document.getElementById("loading");

// Advanced settings elements
const advancedToggleElement = document.getElementById(
	"advanced-toggle",
) as HTMLButtonElement;
const advancedPanelElement = document.getElementById(
	"advanced-panel",
) as HTMLDivElement;
const maxDisparitySliderElement = document.getElementById(
	"max-disparity-slider",
) as HTMLInputElement;
const maxDisparityInputElement = document.getElementById(
	"max-disparity-input",
) as HTMLInputElement;
const maxZoomSliderElement = document.getElementById(
	"max-zoom-slider",
) as HTMLInputElement;
const maxZoomInputElement = document.getElementById(
	"max-zoom-input",
) as HTMLInputElement;
const distanceSliderElement = document.getElementById(
	"distance-slider",
) as HTMLInputElement;
const distanceInputElement = document.getElementById(
	"distance-input",
) as HTMLInputElement;
const numStepsSliderElement = document.getElementById(
	"num-steps-slider",
) as HTMLInputElement;
const numStepsInputElement = document.getElementById(
	"num-steps-input",
) as HTMLInputElement;
const numRepeatsSliderElement = document.getElementById(
	"num-repeats-slider",
) as HTMLInputElement;
const numRepeatsInputElement = document.getElementById(
	"num-repeats-input",
) as HTMLInputElement;
const resetParamsButtonElement = document.getElementById(
	"reset-params-btn",
) as HTMLButtonElement;

if (!containerElement) {
	throw new Error("Canvas container not found");
}

console.log("[main] Initializing GaussianViewer...");

// Initialize viewer
const viewer = new GaussianViewer({
	container: containerElement,
	onLoad: () => {
		console.log("[main] Splat loaded successfully");
		hideLoading();
		enableControls();
	},
	onError: (error) => {
		console.error("[main] Failed to load splat:", error);
		hideLoading();
		alert(`Failed to load PLY file: ${error.message}`);
	},
	onTrajectoryStateChange: (state) => {
		console.log("[main] Trajectory state changed:", state);
		updateButtonStates(state);
	},
	onFrameChange: (_frame, _total) => {
		// Could add a progress indicator here
	},
	// Canvas stays fixed size - splat renders with empty space around it as needed
});

console.log("[main] GaussianViewer initialized");

// UI State Management
function showLoading(): void {
	loadingElement?.classList.add("visible");
}

function hideLoading(): void {
	loadingElement?.classList.remove("visible");
}

function setParameterControlsDisabled(disabled: boolean): void {
	advancedToggleElement.disabled = disabled;
	maxDisparitySliderElement.disabled = disabled;
	maxDisparityInputElement.disabled = disabled;
	maxZoomSliderElement.disabled = disabled;
	maxZoomInputElement.disabled = disabled;
	distanceSliderElement.disabled = disabled;
	distanceInputElement.disabled = disabled;
	numStepsSliderElement.disabled = disabled;
	numStepsInputElement.disabled = disabled;
	numRepeatsSliderElement.disabled = disabled;
	numRepeatsInputElement.disabled = disabled;
	resetParamsButtonElement.disabled = disabled;
}

function enableControls(): void {
	trajectorySelectElement.disabled = false;
	playButtonElement.disabled = false;
	pauseButtonElement.disabled = false;
	resetButtonElement.disabled = false;
	setParameterControlsDisabled(false);
}

function disableControls(): void {
	trajectorySelectElement.disabled = true;
	playButtonElement.disabled = true;
	pauseButtonElement.disabled = true;
	resetButtonElement.disabled = true;
	setParameterControlsDisabled(true);
}

function updateButtonStates(state: "stopped" | "playing" | "paused"): void {
	playButtonElement.disabled = state === "playing";
	pauseButtonElement.disabled = state !== "playing";
	// Disable parameter controls during playback
	setParameterControlsDisabled(state === "playing");
}

// File Loading
async function loadFile(file: File): Promise<void> {
	console.log("[main] loadFile called with:", file.name, file.size, "bytes");

	if (!file.name.toLowerCase().endsWith(".ply")) {
		alert("Please select a PLY file");
		return;
	}

	showLoading();
	disableControls();

	try {
		console.log("[main] Calling viewer.loadPly...");
		await viewer.loadPly(file);
		console.log("[main] viewer.loadPly completed");
	} catch (error) {
		console.error("[main] loadFile error:", error);
		// Error already handled in viewer's onError callback
	}
}

// Event Listeners
fileInputElement?.addEventListener("change", (event) => {
	const target = event.target as HTMLInputElement;
	const file = target.files?.[0];
	if (file) {
		loadFile(file);
	}
});

fileLoaderElement?.addEventListener("click", () => {
	fileInputElement?.click();
});

// Drag and drop
fileLoaderElement?.addEventListener("dragover", (event) => {
	event.preventDefault();
	fileLoaderElement.classList.add("drag-over");
});

fileLoaderElement?.addEventListener("dragleave", () => {
	fileLoaderElement.classList.remove("drag-over");
});

fileLoaderElement?.addEventListener("drop", (event) => {
	event.preventDefault();
	fileLoaderElement.classList.remove("drag-over");

	const file = event.dataTransfer?.files[0];
	if (file) {
		loadFile(file);
	}
});

// Trajectory controls
trajectorySelectElement?.addEventListener("change", () => {
	const type = trajectorySelectElement.value as TrajectoryType;
	viewer.setTrajectoryType(type);
});

playButtonElement?.addEventListener("click", () => {
	viewer.play();
});

pauseButtonElement?.addEventListener("click", () => {
	viewer.pause();
});

resetButtonElement?.addEventListener("click", () => {
	viewer.reset();
});

// Advanced settings toggle
advancedToggleElement?.addEventListener("click", () => {
	const isExpanded = advancedToggleElement.classList.toggle("expanded");
	advancedPanelElement?.classList.toggle("collapsed", !isExpanded);
});

// Helper to sync slider and input values
function syncSliderAndInput(
	slider: HTMLInputElement,
	input: HTMLInputElement,
	onChange: (value: number) => void,
): void {
	slider.addEventListener("input", () => {
		input.value = slider.value;
		onChange(Number.parseFloat(slider.value));
	});

	input.addEventListener("input", () => {
		const value = Number.parseFloat(input.value);
		if (!Number.isNaN(value)) {
			const min = Number.parseFloat(slider.min);
			const max = Number.parseFloat(slider.max);
			const clampedValue = Math.max(min, Math.min(max, value));
			slider.value = String(clampedValue);
			onChange(clampedValue);
		}
	});

	input.addEventListener("blur", () => {
		const value = Number.parseFloat(input.value);
		const min = Number.parseFloat(slider.min);
		const max = Number.parseFloat(slider.max);
		const clampedValue = Math.max(
			min,
			Math.min(
				max,
				Number.isNaN(value) ? Number.parseFloat(slider.value) : value,
			),
		);
		input.value = String(clampedValue);
		slider.value = String(clampedValue);
	});
}

// Wire up parameter controls
syncSliderAndInput(
	maxDisparitySliderElement,
	maxDisparityInputElement,
	(value) => viewer.updateTrajectoryParam("maxDisparity", value),
);

syncSliderAndInput(maxZoomSliderElement, maxZoomInputElement, (value) =>
	viewer.updateTrajectoryParam("maxZoom", value),
);

syncSliderAndInput(distanceSliderElement, distanceInputElement, (value) =>
	viewer.updateTrajectoryParam("distanceMeters", value),
);

syncSliderAndInput(numStepsSliderElement, numStepsInputElement, (value) =>
	viewer.updateTrajectoryParam("numSteps", Math.round(value)),
);

syncSliderAndInput(numRepeatsSliderElement, numRepeatsInputElement, (value) =>
	viewer.updateTrajectoryParam("numRepeats", Math.round(value)),
);

// Reset to defaults
function updateParameterInputsFromDefaults(): void {
	const defaults = DEFAULT_TRAJECTORY_PARAMS;
	maxDisparitySliderElement.value = String(defaults.maxDisparity);
	maxDisparityInputElement.value = String(defaults.maxDisparity);
	maxZoomSliderElement.value = String(defaults.maxZoom);
	maxZoomInputElement.value = String(defaults.maxZoom);
	distanceSliderElement.value = String(defaults.distanceMeters);
	distanceInputElement.value = String(defaults.distanceMeters);
	numStepsSliderElement.value = String(defaults.numSteps);
	numStepsInputElement.value = String(defaults.numSteps);
	numRepeatsSliderElement.value = String(defaults.numRepeats);
	numRepeatsInputElement.value = String(defaults.numRepeats);
}

resetParamsButtonElement?.addEventListener("click", () => {
	viewer.resetTrajectoryParams();
	updateParameterInputsFromDefaults();
});

// Keyboard shortcuts
document.addEventListener("keydown", (event) => {
	if (!viewer.isLoaded()) return;

	switch (event.key) {
		case " ":
			event.preventDefault();
			if (viewer.getPlayerState() === "playing") {
				viewer.pause();
			} else {
				viewer.play();
			}
			break;
		case "r":
		case "R":
			viewer.reset();
			break;
		case "Escape":
			viewer.stop();
			break;
	}
});

// Check for URL parameter to auto-load a file
const urlParams = new URLSearchParams(window.location.search);
const fileUrl = urlParams.get("file");

if (fileUrl) {
	showLoading();
	fetch(fileUrl)
		.then((response) => {
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			return response.blob();
		})
		.then((blob) => {
			const file = new File([blob], "scene.ply", {
				type: "application/octet-stream",
			});
			return loadFile(file);
		})
		.catch((error) => {
			hideLoading();
			console.error("Failed to load file from URL:", error);
		});
}

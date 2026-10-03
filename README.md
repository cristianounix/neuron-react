# Neuron React 

This is a react application to show how a artificial neuron work.

We have:
- Input signals 
- Weights 
- Bias
- Activation function 
- Output signal 

We have a graph in the right side to show how data area distributed.
In the left side we have the neuron structure with the inputs.

Add a button "predict" to interate.
Add animation in the plot when predict.

## Running locally

### Prerequisites

- [Node.js](https://nodejs.org/) 18 or newer (developed with Node 20)
- npm (comes with Node.js)

### Steps

1. Clone the repository and enter the project folder:

   ```bash
   git clone <repository-url>
   cd neuron-react
   ```

2. Install the dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open the URL printed in the terminal (by default http://localhost:5173) in your browser.

5. Click **Predict** to feed the next sample through the neuron, or **Auto** to keep training continuously.

### Other scripts

| Command           | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `npm run dev`     | Start the Vite dev server with hot reload            |
| `npm run build`   | Type-check and build for production into `dist/`     |
| `npm run preview` | Serve the production build locally for a final check |

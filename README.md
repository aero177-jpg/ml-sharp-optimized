> **This is a fork of [Apple's ml-sharp](https://github.com/apple/ml-sharp) repository.**
>

## 🚀 One-Click Cloud Deployment

Deploy the FastAPI endpoint to your own Modal account using GitHub Actions.

### Prerequisites

1. Create a Modal token in the dashboard by visiting `https://modal.com/settings/<your-username>/tokens`, clicking “Create new”, and copying the token ID and secret from the generated string.

Alternatively, you can authenticate locally once with:

```bash
uv run modal token new
```

2. Go to `sharp-api-auth`:  
   `https://modal.com/secrets/<your-username>/main/create?secret_name=sharp-api-auth`   and create a key value secret. Remember the value, this will be used in the frontend client.

2. Add the following GitHub Actions secrets in your fork:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

### Deploy

1. Go to your repo’s Actions tab.
2. Run the “Deploy Modal API” workflow.
3. Find the FastAPI URL in your Modal dashboard after deploy.

The workflow deploys [src/sharp/modal/api.py](src/sharp/modal/api.py) to Modal.

---

## Cloud GPU Inference (No Local GPU Required)

Generate Gaussian splats using [Modal](https://modal.com)'s cloud GPUs. Modal offers a free tier with $30/month in credits.

### Setup (One-Time)

```bash
# Install with cloud support
uv pip install -e ".[cloud]"

# Create Modal account and authenticate
uv run modal token new
```

### Optional: Pre-warm Model Cache

```bash
uv run sharp cloud setup
```

This downloads the ~800MB model to Modal's cloud storage, making subsequent runs faster.

### Usage

```bash
# Run inference on cloud GPU (default: A10 @ $1.10/hr)
uv run sharp cloud predict -i photo.jpg -o output/

# Process multiple images
uv run sharp cloud predict -i photos/ -o output/

# Choose GPU tier
uv run sharp cloud predict -i photo.jpg -o output/ --gpu t4    # $0.59/hr (budget)
uv run sharp cloud predict -i photo.jpg -o output/ --gpu h100  # $3.95/hr (fastest)
```

### Available GPU Tiers

| GPU  | Price/hr | Notes             |
| ---- | -------- | ----------------- |
| T4   | $0.59    | Budget option     |
| L4   | $0.80    | Good value        |
| A10  | $1.10    | Default, balanced |
| A100 | $2.50    | High performance  |
| H100 | $3.95    | Fastest           |



## Original README

For more information visit https://github.com/apple/ml-sharp.

## Credits

Credits to [diegocodez](https://github.com/diegocodez) for sog format support, and [kstonekuan](https://github.com/kstonekuan) for Modal integration.

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.

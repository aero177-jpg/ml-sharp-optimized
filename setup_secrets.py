"""CLI helper to create Modal secrets for ml-sharp."""

from __future__ import annotations

import modal

SUPABASE_SECRET_NAME = "supabase-creds"
API_AUTH_SECRET_NAME = "sharp-api-auth"


def setup() -> None:
    print("🚀 Setting up Modal Secrets for ml-sharp-optimized...")

    print("\n--- Supabase Setup ---")
    url = input("Enter Supabase URL: ").strip()
    key = input("Enter Supabase Service Role Key: ").strip()
    bucket = input("Enter Supabase Bucket Name [default: testbucket]: ").strip() or "testbucket"

    modal.Secret.objects.create(
        name=SUPABASE_SECRET_NAME,
        env_dict={
            "SUPABASE_URL": url,
            "SUPABASE_KEY": key,
            "SUPABASE_BUCKET": bucket,
        },
        allow_existing=True,
    )

    print("\n--- API Security ---")
    auth_token = input("Create a secret API Key (for X-API-KEY header): ").strip()

    modal.Secret.objects.create(
        name=API_AUTH_SECRET_NAME,
        env_dict={"API_AUTH_TOKEN": auth_token},
        allow_existing=True,
    )

    print("\n✅ Secrets successfully created on Modal.com!")
    print("You can now run the GitHub Action to deploy.")


if __name__ == "__main__":
    setup()

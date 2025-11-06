import os
import uvicorn

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

import server.app

# Load environment variables

HOST = os.getenv("ROTARY_INSIGHT_HOST", "0.0.0.0")
PORT = int(os.getenv("ROTARY_INSIGHT_PORT", "8000"))


def main():
    """Run the server."""
    uvicorn.run(
        "server.app:app",
        host=HOST,
        port=PORT,
        reload=False,  # Set to True for development
        log_level="info",
    )


if __name__ == "__main__":
    main()

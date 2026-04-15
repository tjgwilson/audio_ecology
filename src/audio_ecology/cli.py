import typer

app = typer.Typer()


@app.command()
def hello() -> None:
    """Simple test command."""
    print("Audio ecology pipeline ready.")


if __name__ == "__main__":
    app()
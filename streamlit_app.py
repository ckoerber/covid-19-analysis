"""
"""
from streamlit import markdown, sidebar, number_input
from os import path

ROOT = path.abspath(path.dirname(__file__))


def get_md(file):
    """Returns markdown for file name.
    """
    with open(path.join(ROOT, "doc", file), "r") as inp:
        content = inp.read()
    return markdown(content)


def create_sidebar(model):
    """
    """
    elements = dict()
    for i_name, i_data in model:
        elements[i_name] = sidebar.number_input(**input)

    return elements


def main():
    get_md("intro.md")


if __name__ == "__main__":
    main()
    create_sidebar()

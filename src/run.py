from argparse import ArgumentParser
from os import environ

from data.tagging import gen_tags
from parsers.sum_parser import add_summarization_parser
from parsers.tagging_parser import add_tagging_args
from summarization import semantic_sum

environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # only print warnings and errors


def main() -> None:
    parser = ArgumentParser(
        description="Module for running scripts within the repo.",
    )
    subparsers = parser.add_subparsers()

    parser_tagging = subparsers.add_parser(
        "tag",
        help="Generate tags from episodes",
    )
    add_tagging_args(parser_tagging)
    parser_tagging.set_defaults(main=gen_tags.main)

    parser_summarization = subparsers.add_parser(
        "summarize",
        help="Generate summaries from episode tags",
    )
    add_summarization_parser(parser_summarization)
    parser_summarization.set_defaults(main=semantic_sum.main)

    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()

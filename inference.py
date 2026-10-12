"""Generate videos using a hierarchical checkpoint."""

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv or "-h" in sys.argv or "--print-config" in sys.argv:
        from multics.inference_args import parse_args

        parse_args()
    else:
        from multics.inference import main

        main()

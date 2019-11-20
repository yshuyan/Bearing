import argparse

parser = argparse.ArgumentParser()
flag_parser = parser.add_mutually_exclusive_group(required=False)
flag_parser.add_argument('--flag', dest='flag', action='store_true')
flag_parser.add_argument('--no-flag', dest='flag', action='store_false')
parser.set_defaults(flag=True)

args = parser.parse_args()

print(args.flag)
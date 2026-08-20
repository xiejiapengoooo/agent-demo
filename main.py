def run():
    from uuid import NAMESPACE_URL, uuid5
    print(str(uuid5(NAMESPACE_URL, "0000")))


if __name__ == "__main__":
    run()

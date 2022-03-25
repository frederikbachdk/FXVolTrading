def enumerate2(xs, start=0, step=2):
    for x in xs:
        yield (start, x)
        start += step
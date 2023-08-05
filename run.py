#!/usr/bin/env python3

from roop import core

from inference_codeformer import inference_codeformer


if __name__ == "__main__":
    newface = "/root/Desktop/save/3.png"
    oldface = "/root/Desktop/save/4.png"
    savepath = "outputs/output.png"

    core.run(newface, oldface, savepath)
    inference_codeformer.run(savepath)

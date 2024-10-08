#!/usr/bin/env python3

import os
import argparse

from pathlib import Path
import pprint

from rosbags.highlevel import AnyReader


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Extract data from requested topics in an mcap recording and save them to the filesystem."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input bag path (folder or filepath) to read from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for serialized data",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--topics",
        metavar="T",
        type=str,
        nargs="+",
        help="List of topics to be serialized",
        required=False,
    )
    args = parser.parse_args()
    return args


def get_friendly_topic_name(topic):
    ''' Generates a topic name that doesn't contain forward slashes '''
    return topic.replace("/", "_")[1:]

def print_topics():
    args = parse_cli_args()

    topics = []
    with AnyReader([Path(args.input)]) as reader:
        connections = [x for x in reader.connections]        
        for connection, _, rawdata in reader.messages(connections=connections):
            topics.append(connection.topic)

    pprint.pprint(set(topics))
    return

def extract_from_rosbag(args, filepath):
    print(f"Extracting rosbag {filepath}")
    with AnyReader([Path(filepath)]) as reader:
        connections = [x for x in reader.connections if x.topic in args.topics]
        for connection, _, rawdata in reader.messages(connections=connections):
            rosbag_topics_output_path = os.path.join(args.output, os.path.basename(filepath)[:-4], get_friendly_topic_name(connection.topic))
            os.makedirs(rosbag_topics_output_path, exist_ok=True)
            if "image_data" in rosbag_topics_output_path:
                msg = reader.deserialize(rawdata[:-4], connection.msgtype)
                prefix = connection.msgtype.split('/')[-1].lower()
                timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                separator = "_"
                resolution = f"{msg.width}x{msg.height}"
                encoding = msg.encoding.lower()
                step = f"step_{msg.step}"
                ext = ".bin"

                of = os.path.join(rosbag_topics_output_path, f"{prefix}{separator}{timestamp}{separator}{resolution}{separator}{step}{separator}{encoding}{ext}")
                with open(of, "w+b") as f:
                    f.write(msg.data)

            elif "info_camera_info" in rosbag_topics_output_path:
                msg = reader.deserialize(rawdata, connection.msgtype)
                prefix = "info"
                separator = "_"
                ext = ".txt"

                of = os.path.join(rosbag_topics_output_path, f"{prefix}{ext}")
                with open(of, "w") as f:
                    for i in range(9):
                        f.write(f"K[{i}]={msg.K[i]}\n")
            
            else:
                print(f"I don't know how to parse topic: {connection.topic}")


def main():
    args = parse_cli_args()

    inputPath = args.input
    if os.path.isdir(inputPath):
        for name in os.listdir(inputPath):
            extract_from_rosbag(args, os.path.join(inputPath, name))
    else:
        extract_from_rosbag(args, args.input)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import json
import typing
import pickle

class HectorNVTXEvent:
    def __init__(self, json_data: dict):
        self.json_data = json_data
        if not "Timestamp" in self.json_data["NvtxEvent"]:
            print("Warning: This record does not have a timestamp")
            print(json_data)
            self.start_timestamp = -1
        else:
            self.start_timestamp = int(json_data["NvtxEvent"]["Timestamp"])
        if not "EndTimestamp" in self.json_data["NvtxEvent"]:
            print("Warning: This record does not have an end timestamp")
            print(json_data)
            self.end_timestamp = -1
        else:
            self.end_timestamp = int(json_data["NvtxEvent"]["EndTimestamp"])
        if not "GlobalTid" in self.json_data["NvtxEvent"]:
            print("Warning: This record does not have a global tid")
            print(json_data)
            self.global_tid = -1
        else:
            self.global_tid = int(json_data["NvtxEvent"]["GlobalTid"])
        if not json_data["NvtxEvent"]["NsTime"]:
            print("Warning: This record is not using nanosecond")
            print(json_data)

        # extract seq and op_id if there are any
        self.seq = -1
        self.op_id = -1
        self.hector_op_category = ""
        self.text = ""
        if not "Text" in self.json_data["NvtxEvent"]:
            print("Warning: This record does not have a text")
            print(json_data)
        else:
            self.text = json_data["NvtxEvent"]["Text"]
            text_parts = self.text.split(",")
            for text_part in text_parts:
                if "seq" in text_part:
                    self.seq = int(text_part.split(" = ")[1])
                if "op_id" in text_part:
                    self.op_id = int(text_part.split(" = ")[1])
                if "hector_op_category" in text_part:
                    self.hector_op_category = text_part.split(" = ")[1].strip()
        self.children = []

    def insert_child(self, nvtx_event):
        insert_child(self.children, nvtx_event)

    def yield_events(self):
        yield self
        for child in self.children:
            yield from child.yield_events()

    def _get_seq_category(self, category=""):
        result = dict()
        for child in self.children:
            if len(child.hector_op_category) !=0:
                if len(category) != 0:
                    print("Warning: category is not empty", category, child.hector_op_category, child.text)
                    category = category+";"+child.hector_op_category
                else:
                    category = child.hector_op_category
            result.update(child._get_seq_category(category))
        if self.seq != -1 and len(category)!=0:
            result[self.seq] = category
        return result

    def get_seq_category(self):
        return self._get_seq_category(self.hector_op_category)

    def filter_events(self, start_timestamp, end_timestamp):
        if self.start_timestamp > end_timestamp or self.end_timestamp < start_timestamp:
            # no overlap
            return None
        if self.start_timestamp >= start_timestamp and self.end_timestamp <= end_timestamp:
            # complete overlap
            return [self]
        result = []
        for child in self.children:
            curr_result = child.filter_events(start_timestamp, end_timestamp)
            result += curr_result
        return self

def get_parent_in_pairs(event_a: HectorNVTXEvent, event_b: HectorNVTXEvent) -> str:
    if event_a.start_timestamp == event_b.start_timestamp and event_a.end_timestamp == event_b.end_timestamp:
        print("Warning: complete overlap assuming a is parent")
        print(event_a.text)
        print(event_b.text)
        return "event a is parent"
    if event_a.start_timestamp < event_b.start_timestamp and event_a.end_timestamp > event_b.end_timestamp:
        #return event_a
        return "event a is parent"
    if event_b.start_timestamp < event_a.start_timestamp and event_b.end_timestamp >= event_a.end_timestamp:
        #return event_b
        return "event b is parent"
    if (event_a.end_timestamp < event_b.start_timestamp) or  (event_a.start_timestamp > event_b.end_timestamp):
        #return event_a
        return "no overlap"
    print("Warning: partial overlap. assuming no overlap as this should only happen in different threads")
    print(event_a.text)
    print(event_b.text)
    return "no overlap"

def insert_child(children, nvtx_event):
    # recursively insert the child to the lowest level
    for child in sorted(children, key=lambda x: x.start_timestamp):
        if get_parent_in_pairs(child, nvtx_event) == "event a is parent":
            child.insert_child(nvtx_event)
            return
        elif get_parent_in_pairs(child, nvtx_event) == "event b is parent":
            nvtx_event.children.append(child)
            children.remove(child)
            children.append(nvtx_event)
            return
    children.append(nvtx_event)


# This is effectively a dummy HectorNVTXEvent node where all its children recursively store the events in a thread
class HectorNVTXEventTree:
    def __init__(self, global_tid):
        #, root: typing.Union[HectorNVTXEvent, None] = None
        self.children = [] # events
        # self.children_startTimestamp = dict()# {startTimestamp: event}
        # self.children_endTimestamp = dict()# {endTimestamp: event}
        self.global_tid = global_tid

    def insert_child(self, nvtx_event):
        insert_child(self.children, nvtx_event)

    def build_tree(self, nvtx_events: typing.List[HectorNVTXEvent]):
        # the input is a list of HedtorNVTXEvents instead of dictionary of {startTimestamp: event} to avoid the case where two events have the same start timestamp
        for nvtx_event in nvtx_events:
            self.insert_child(nvtx_event)

    def yield_events(self):
        for child in self.children:
            yield from child.yield_events()

    def get_seq_category(self):
        result = dict()
        for child in self.children:
            result.update(child.get_seq_category())
        return result

    def filter_events(self, start_timestamp, end_timestamp):
        result = []
        for child in self.children:
            curr_result = child.filter_events(start_timestamp, end_timestamp)
            result += curr_result

        # create a HectorNVTXEventTree object to store the result
        result_tree = HectorNVTXEventTree(self.global_tid)
        result_tree.children = result
        return result_tree
    def save_to_file(self, dump_file_name):
        with open(dump_file_name, 'wb') as fd:
            pickle.dump(self,fd)

class HectorNVTXEventsPerProcess:
    def __init__(self, nvtx_trees: typing.List[HectorNVTXEventTree]):
        self.nvtx_trees = {nvtx_tree.global_tid: nvtx_tree for nvtx_tree in nvtx_trees}
        self.nvtx_trees = dict() # {tid: HectorNVTXEventTree}
        self.seq_id_category = dict() #{seq: category}
        self.forward_prop_periods = set() # {(start_timestamp, end_timestamp)}
        self.backward_prop_periods = set() # {(start_timestamp, end_timestamp)}
        self.optimizer_step_periods = set() # {(start_timestamp, end_timestamp)}
        self.benchmark_record_periods = dict() # {bench_name: (start_timestamp, end_timestamp)}
    def yield_events(self):
        for nvtx_tree in self.nvtx_trees.values():
            yield from nvtx_tree.yield_events()
    def extract_forward_prop_periods(self):
        for event in self.yield_events():
            if event.text.find("hector_forward_mark") != -1:
                self.forward_prop_periods.add((event.start_timestamp, event.end_timestamp))
    def extract_backward_prop_periods(self):
        for event in self.yield_events():
            if event.text.find("hector_backward_mark") != -1:
                self.backward_prop_periods.add((event.start_timestamp, event.end_timestamp))
    def extract_optimizer_step_periods(self):
        for event in self.yield_events():
            if event.text.find("hector_optimizer_step") != -1:
                self.optimizer_step_periods.add((event.start_timestamp, event.end_timestamp))
    def extract_benchmark_record_periods(self):
        for event in self.yield_events():
            if event.text.find("hector_benchmark_record") != -1:
                self.benchmark_record_periods[event.text] = (event.start_timestamp, event.end_timestamp)
    def propagate_seq_id(self):
        # TODO:
        raise NotImplementedError
        pass
        # for nvtx_tree in self.nvtx_trees.values():
        #     self.seq_id_category.update(nvtx_tree.get_seq_category())
    def extract_seq_id_category(self):
        for nvtx_tree in self.nvtx_trees.values():
            self.seq_id_category.update(nvtx_tree.get_seq_category())
    def filter_events(self, start_timestamp, end_timestamp):
        # return new HectorNVTXEventsPerProcess
        result_trees = []
        for nvtx_tree in self.nvtx_trees.values():
            curr_result = nvtx_tree.filter_events(start_timestamp, end_timestamp)
            result_trees.append(curr_result)
        result_process = HectorNVTXEventsPerProcess(result_trees)
        result_process.seq_id_category = self.seq_id_category
        result_process.forward_prop_periods = self.forward_prop_periods
        result_process.backward_prop_periods = self.backward_prop_periods
        result_process.benchmark_record_periods = self.benchmark_record_periods
        return result_process


def is_json_nvtx_event(json_data):
    # NB: NVTX event types are defined in https://docs.nvidia.com/nsight-systems/UserGuide/index.html
    # """
    # NVTX Event Type Values
    # 33 - NvtxCategory
    # 34 - NvtxMark
    # 39 - NvtxThread
    # 59 - NvtxPushPopRange
    # 60 - NvtxStartEndRange
    # 75 - NvtxDomainCreate
    # 76 - NvtxDomainDestroy
    # """
    result = "NvtxEvent" in json_data
    if result:
        if not json_data["Type"] == 59:
            print(json_data["Type"])
            result = False
        #assert(json_data["Type"] == 59)
    return result

def retrieve_nvtx_events_from_nsys_json(json_data_filename):
    nvtx_events = dict() # {tid: dict[starttimestamp:HectorNVTXEvent]}
    with open(json_data_filename) as fd:
        for line in fd:
            line_data = json.loads(line)
            if is_json_nvtx_event(line_data):
                curr_event = HectorNVTXEvent(line_data)
                if curr_event.global_tid not in nvtx_events:
                    #nvtx_events[curr_event.global_tid] = dict()
                    nvtx_events[curr_event.global_tid] = []
                #nvtx_events[curr_event.global_tid][curr_event.start_timestamp] = curr_event
                nvtx_events[curr_event.global_tid].append(curr_event)
    for tid in nvtx_events:
        event_tree_for_curr_tid = HectorNVTXEventTree(tid)
        event_tree_for_curr_tid.build_tree(nvtx_events[tid])
        nvtx_events[tid] = event_tree_for_curr_tid
    return nvtx_events

if __name__ == "__main__":
    nvtx_events = retrieve_nvtx_events_from_nsys_json("hgt_acm.json")

    nvtx_events = []
    with open("hgt_acm.json") as fd:
        result = 0
        for line in fd:
            if "NvtxEvent" in line:
                nvtx_events.append(json.loads(line))
                result += is_json_nvtx_event(nvtx_events[-1])
        print(result)

# A multi-level NVTX range looks like this:
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739751477","Text":"autograd::engine::evaluate_function: AddmmBackward0, op_id = 1718","GlobalTid":"281863302175408","EndTimestamp":"4740091747","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739753014","Text":"AddmmBackward0, seq = 596, op_id = 1719","GlobalTid":"281863302175408","EndTimestamp":"4740063789","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739759015","Text":"aten::t, op_id = 1720","GlobalTid":"281863302175408","EndTimestamp":"4739762864","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739760324","Text":"aten::transpose, op_id = 1721","GlobalTid":"281863302175408","EndTimestamp":"4739762002","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739761055","Text":"aten::as_strided, op_id = 1722","GlobalTid":"281863302175408","EndTimestamp":"4739761828","DomainId":"0","NsTime":true}}
# {"Type":59,"NvtxEvent":{"Type":59,"Timestamp":"4739764299","Text":"aten::mm, op_id = 1723","GlobalTid":"281863302175408","EndTimestamp":"4739988492","DomainId":"0","NsTime":true}}

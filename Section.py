import copy
import AudioUtil as AudioUtil

class Section:
    def __init__(self, args):
        self.start = args.get("start")
        self.end = args.get("end")
        self.start_sample = args.get("start_sample", 0)
        self.end_sample = args.get("end_sample", 0)
        
        self.confidence = args.get("confidence", 1)
        self.group_id = args.get("group_id", 0)
        self.key = args.get("key", -1)
        self.mds_feature = args.get("mds_feature", None)
        
        # family variables
        self.parent = False
        self.path_family = None
        self.path_family_scores = None
        self.score = None
        self.normalized_score = None
        self.coverage = None
        self.normalized_coverage = None
        self.fitness = None

        # Additional properties from the constructor
        self.color_angle = args.get("color_angle", None)
        self.color_radius = args.get("color_radius", None)

    def get_duration(self):
        return self.end - self.start

    def overlaps(self, other_section):
        return (
            (self.start <= other_section.start and self.end > other_section.start) or
            (self.start < other_section.end and self.end >= other_section.end) or
            (other_section.start <= self.start and other_section.end > self.start) or
            (other_section.start < self.start and other_section.end >= self.end) or
            (self.start >= other_section.start and self.end <= other_section.end) or
            (other_section.start >= self.start and other_section.end <= self.end)
        )

    def splits(self, other_section):
        return other_section.start < self.start and other_section.end > self.end

    def covers(self, other_section):
        return self.start <= other_section.start and self.end >= other_section.end

    def clips_start(self, other_section):
        return self.end > other_section.start and self.end < other_section.end and self.start <= other_section.start

    def clips_end(self, other_section):
        return self.start > other_section.start and self.start < other_section.end and self.end >= other_section.end

    def disjoint(self, other_section):
        return self.end <= other_section.start or self.start >= other_section.end

    def clone(self):
        return copy.deepcopy(self)

    def amount_of_sections_after_subtract(self, other_section):
        if other_section.covers(self):
            return 0
        if other_section.splits(self):
            return 2
        return 1

    def subtract(self, other_section):
        if other_section.disjoint(self):
            return self

        assert (
            self.amount_of_sections_after_subtract(other_section) == 1,
            f"Won't subtract, subtraction yields {self.amount_of_sections_after_subtract(other_section)} sections [{self.start}, {self.end}] - [{other_section.start}, {other_section.end}]"
        )

        if other_section.clips_end(self):
            self.end = other_section.start
        elif other_section.clips_start(self):
            self.start = other_section.end

        if self.end <= self.start:
            raise ValueError("Section end is before start")

        return self

    def subtract_and_create_new(self, other_section):
        if other_section.covers(self):
            return []
        if other_section.splits(self):
            left = self.clone()
            left.end = other_section.start
            right = self.clone()
            right.start = other_section.end
            return [left, right]

        this_clone = self.clone()
        return [this_clone.subtract(other_section)]

    def get_key_name(self):
        return AudioUtil.key_names[self.key] 

from SpotifySegment import SpotifySegment
import logging as log
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math

class Features:
    def __init__(self, analysis_data, options=None):
        if options is None:
            options = {}

        self.duration = 0
        self.length = 0
        self.segments = []
        self.segment_start_duration = []
        self.raw = {
            'pitches': [],
            'loudness': []
        }
        self.processed = {
            'pitches': [],
            'noise': [],
            'loudness': [],
            'avg_loudness': [],
            'dynamics': [],
            'tonal_energy': [],
            'tonal_radius': [],
            'tonal_angle': []
        }
        self.cluster_selection = []
        self.tsne_selection = []
        self.sample_duration = 0
        self.sample_amount = 0
        self.sampled = {}
        self.sample_start_duration = []
        self.sample_blur = 0  # in proportion to duration (<1 is no blur, 2 is blur of twice duration)
        self.fast_sampled_pitch = []
        self.fast_sample_duration = 0.1
        self.direct_loudness_sample_duration = 0.25
        self.direct_loudness_amount = None
        self.direct_loudness = None
        self.max_loudness = None
        self.average_loudness = None
        self.dynamics_base = 0.12

        self.duration = analysis_data['track']['duration']
        for segment in analysis_data['segments']:
            self.segments.append(SpotifySegment(segment))
            self.segment_start_duration.append([segment['start'], segment['duration']])
        self.length = len(self.segments)
        self.sample_amount = min(math.ceil(self.duration / options.get('sample_duration', 1)), options.get('samples', 0))
        self.sample_duration = analysis_data['track']['duration'] / self.sample_amount

        self.sample_blur = options.get('sample_blur', 1)
        
        print(f"Sampling, Amount: {self.sample_amount}, Duration: {self.sample_duration}")

        self.process_segments()
        if len(self.segments):
            self.process_direct_loudness()

        self.sample_features()
        self.process_samples()
        self.fast_sampled_pitch = self.sample("pitches", {"sample_duration": self.fast_sample_duration})

    def process_segments(self):
        for i, s in enumerate(self.segments):
            self.raw['pitches'].append(s.segment['pitches'])
            self.raw['loudness'].append(s.get_loudness_features())
            s.process_pitch()

            self.processed['pitches'].append(s.pitches)
            self.processed['noise'].append(s.noise)
            self.processed['loudness'].append(s.get_loudness_features())

            next_segment_start_loudness = self.segments[i + 1].get_loudness_features()[0] if i + 1 < len(self.segments) else 0
            self.processed['avg_loudness'].append(s.get_average_loudness(next_segment_start_loudness))
            self.processed['dynamics'].append(s.get_average_loudness(next_segment_start_loudness))

            self.processed['tonal_energy'].append(s.tonality_energy)
            self.processed['tonal_radius'].append(s.tonality_radius)
            self.processed['tonal_angle'].append(s.tonality_angle)

    def sample_features(self):
        # Fill sample start duration
        for i in range(self.sample_amount - 1):
            self.sample_start_duration.append([i * self.sample_duration, self.sample_duration])
        
        # Last sample is shorter
        last_sample_start = (self.sample_amount - 1) * self.sample_duration
        self.sample_start_duration.append([last_sample_start, self.duration - last_sample_start])
        self.init_sample_features()

        blur_duration = self.sample_blur * self.sample_duration
        blur_outside_sample_duration = (blur_duration - self.sample_duration) / 2

        for segment_index, segment in enumerate(self.segments):
            segment_end = segment.start + segment.duration

            # Calculate range of samples e.g. [2,6] clip by 0 and size
            sample_range_start_index = max(
                0,
                math.floor((segment.start - blur_outside_sample_duration) / self.sample_duration)
            )
            sample_range_end_index = min(
                self.sample_amount - 1,
                math.floor((segment_end + blur_outside_sample_duration) / self.sample_duration)
            )

            range_size = sample_range_end_index - sample_range_start_index
            if range_size >= 1:
                # First sample in range
                first_sample = self.sample_start_duration[sample_range_start_index]
                sample_blur_end = first_sample[0] + first_sample[1] + blur_outside_sample_duration
                first_sample_overlap = sample_blur_end - segment.start
                self.add_features_scaled(sample_range_start_index, segment_index, first_sample_overlap)
            
            if range_size >= 1:
                # Last sample in range
                sample_blur_start = self.sample_start_duration[sample_range_end_index][0] - blur_outside_sample_duration
                last_sample_overlap = segment_end - sample_blur_start
                self.add_features_scaled(sample_range_end_index, segment_index, last_sample_overlap)
            
            if range_size >= 2:
                # Every middle sample
                for i in range(sample_range_start_index + 1, sample_range_end_index):
                    self.add_features_scaled(i, segment_index, blur_duration)
            
            if range_size == 0:
                self.add_features_scaled(sample_range_start_index, segment_index, segment.duration)

        # First sample has only blur on right
        self.divide_features(0, self.sample_duration + blur_outside_sample_duration)

        # Last sample has shorter duration and only blur on left
        self.divide_features(
            self.sample_amount - 1,
            self.sample_start_duration[self.sample_amount - 1][1] + blur_outside_sample_duration
        )

        for i in range(1, self.sample_amount - 1):
            self.divide_features(i, blur_duration)

    def init_sample_features(self):
        for feature_name, feature_value in self.processed.items():
            self.sampled[feature_name] = [0] * self.sample_amount

            feature_size = len(feature_value[0]) if isinstance(feature_value[0], (list, tuple)) else 0

            if feature_size:
                for s in range(self.sample_amount):
                    self.sampled[feature_name][s] = [0] * feature_size
            else:
                for s in range(self.sample_amount):
                    self.sampled[feature_name][s] = 0

    def add_features_scaled(self, sample_index, segment_index, scalar):
        for feature_name, feature_value in self.processed.items():
            feature_size = len(feature_value[0]) if isinstance(feature_value[0], (list, tuple)) else 0
            
            if feature_size:
                for i in range(feature_size):
                    self.sampled[feature_name][sample_index][i] += self.processed[feature_name][segment_index][i] * scalar
            else:
                self.sampled[feature_name][sample_index] += self.processed[feature_name][segment_index] * scalar

    def divide_features(self, sample_index, divisor):
        for feature_name, feature_value in self.sampled.items():
            feature_size = len(feature_value[0]) if isinstance(feature_value[0], (list, tuple)) else 0
            
            if feature_size:
                for i in range(feature_size):
                    self.sampled[feature_name][sample_index][i] /= divisor
            else:
                self.sampled[feature_name][sample_index] /= divisor

    def process_samples(self):
        self.sampled_smoothed_avg_loudness = gaussian_filter1d(self.sampled['avg_loudness'], 1)
        self.average_loudness = 0
        self.max_loudness = 0
        for loudness in self.sampled_smoothed_avg_loudness:
            self.average_loudness += loudness
            if loudness > self.max_loudness:
                self.max_loudness = loudness

        log.debug("Maxloudness", self.max_loudness)
        log.debug(self.sampled['dynamics'][0])
        log.debug(self.sampled['dynamics'][0] / self.max_loudness)
        log.debug((self.sampled['dynamics'][0] / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base)

        self.processed['dynamics'] = [(dynamic / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base for dynamic in self.processed['dynamics']]
        self.sampled['dynamics'] = [(dynamic / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base for dynamic in self.sampled['dynamics']]
        log.debug(self.sampled['dynamics'][0])

        self.average_loudness /= len(self.sampled_smoothed_avg_loudness)

    def process_direct_loudness(self):
        self.direct_loudness_amount = math.floor(self.duration / self.direct_loudness_sample_duration)
        self.direct_loudness = np.zeros(self.direct_loudness_amount, dtype=np.float32)

        print("Process direct loudness", 
              "duration", self.duration, 
              "segment_amount", len(self.segment_start_duration), 
              "amount", self.direct_loudness_amount)

        segment_index = 0
        for i in range(self.direct_loudness_amount):
            time = self.direct_loudness_sample_duration * i
            while not self.is_time_in_segment(segment_index, time):
                segment_index += 1
            self.direct_loudness[i] = self.get_exact_loudness(segment_index)

    def is_time_in_segment(self, index, time):
        start = self.segment_start_duration[index][0]
        end = start + self.segment_start_duration[index][1]
        return time >= start and time < end

    def get_exact_loudness(self, index):
        return self.raw['loudness'][index][1]

    def sample(self, feature, options):
        sampled_feature = []

        sample_amount = 0
        sample_duration = 0
        if 'sample_duration' in options:
            # make sure the length of the track is divisible by sample_duration
            sample_amount = round(self.duration / options.get("sample_duration"))
            sample_duration = self.duration / sample_amount
        if 'sample_amount' in options:
            sample_amount = options.get("sample_amount")
            sample_duration = self.duration / sample_amount

        i = 0
        for s in range(sample_amount):
            average_feature = np.zeros(12, dtype=np.float32)

            sample_start = s * sample_duration
            sample_end = (s + 1) * sample_duration

            # Sample is contained in segment, simply copy pitch from segment and go to next sample
            if self.segments[i].get_end() > sample_end:
                average_feature = np.add(average_feature, self.segments[i].get_feature_by_name(feature))
                sampled_feature.append(average_feature)
                continue

            # add part of first segment
            if self.segments[i].get_end() > sample_start:
                weight = (self.segments[i].get_end() - sample_start) / sample_duration
                weight_result = [x * weight for x in self.segments[i].get_feature_by_name(feature)]
                average_feature = np.add(average_feature, weight_result)
                i += 1

            # while entire segment is contained in sample
            while i < len(self.segments) and self.segments[i].get_end() < sample_end:
                weight = self.segments[i].duration / sample_duration
                weight_result = [x * weight for x in self.segments[i].get_feature_by_name(feature)]
                average_feature = np.add(average_feature, weight_result)
                i += 1

            # add part of last segment
            if i < len(self.segments):
                weight = (sample_end - self.segments[i].start) / sample_duration
                weight_result = [x * weight for x in self.segments[i].get_feature_by_name(feature)]
                average_feature = np.add(average_feature, weight_result)

            sampled_feature.append(average_feature)

        return sampled_feature
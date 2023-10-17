from SpotifySegment import SpotifySegment
import logging as log
import numpy as np

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
        self.sample_amount = min(int(self.duration / options.get('sample_duration', 1)), options.get('samples', 0))
        self.sample_duration = analysis_data['track']['duration'] / self.sample_amount

        self.sample_blur = options.get('sample_blur', 1)
        
        log.debug(f"Sampling, Amount: {self.sample_amount}, Duration: {self.sample_duration}")

        # Assuming a `process_segments` method exists or will be implemented
        self.process_segments()
        if len(self.segments):
            # Assuming a `process_direct_loudness` method exists or will be implemented
            self.process_direct_loudness()

        # Assuming a `sample_features` method exists or will be implemented
        self.sample_features()
        # Assuming a `process_samples` method exists or will be implemented
        self.process_samples()

        # Assuming a `sample` method exists or will be implemented
        self.fast_sampled_pitch = self.sample("pitches", {"sampleDuration": self.fast_sample_duration})

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
                int((segment.start - blur_outside_sample_duration) / self.sample_duration)
            )
            sample_range_end_index = min(
                self.sample_amount - 1,
                int((segment_end + blur_outside_sample_duration) / self.sample_duration)
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
        pass

    def process_direct_loudness(self):
        pass

    def sample(self, features, options):
        pass
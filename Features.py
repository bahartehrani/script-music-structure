import SpotifySegment
from logging import log
from scipy.ndimage import gaussian_filter

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
            'avgLoudness': [],
            'dynamics': [],
            'tonalEnergy': [],
            'tonalRadius': [],
            'tonalAngle': []
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
        self.sample_amount = min(int(self.duration / options.get('sampleDuration', 1)), options.get('samples', 0))
        self.sample_duration = analysis_data['track']['duration'] / self.sample_amount

        self.sample_blur = options.get('sampleBlur', 1)
        
        logging.info(f"Sampling, Amount: {self.sample_amount}, Duration: {self.sample_duration}")

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
            self.raw['pitches'][i] = s.segment['pitches']
            self.raw['loudness'][i] = s.get_loudness_features()
            ###
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

        for i in range(len(self.segments)):
            if 0 < i < len(self.segments) - 1:
                self.segments[i].process_pitch_smooth(self.segments[i - 1], self.segments[i + 1])

    def process_direct_loudness(self):
        self.direct_loudness_amount = int(self.duration / self.direct_loudness_sample_duration)
        self.direct_loudness = [0.0] * self.direct_loudness_amount

        print("Process direct loudness", 
              "duration", self.duration, 
              "segment_amount", len(self.segment_start_duration), 
              "amount", self.direct_loudness_amount)

        segment_index = 0
        for i in range(self.direct_loudness_amount):
            time = self.direct_loudness_sample_duration * i
            while not self.is_time_in_segment(segment_index, time):
                segment_index += 1
            self.direct_loudness[i] = self.get_exact_loudness(segment_index, time)

    def is_time_in_segment(self, index, time):
        start = self.segment_start_duration[index][0]
        end = start + self.segment_start_duration[index][1]
        return time >= start and time < end

    def get_exact_loudness(self, index, time):
        return self.raw['loudness'][index][1]

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

    def process_samples(self):
        self.sampled_chords = []
        self.sampled_majorminor = []
        for i in range(self.sample_amount):
            # Assuming chord_detection is a global object or module
            self.sampled_chords.append(chord_detection.get_pop_chord(self.sampled_pitches[i]))
            self.sampled_majorminor.append(chord_detection.get_major_minor_ness(self.sampled_pitches[i]))

        self.sampled_smoothed_avg_loudness = gaussian_filter(self.sampled_avg_loudness, 1)
        self.average_loudness = 0
        self.max_loudness = 0
        for loudness in self.sampled_smoothed_avg_loudness:
            self.average_loudness += loudness
            if loudness > self.max_loudness:
                self.max_loudness = loudness

        # Assuming log is a global object or module for logging
        log.debug("Maxloudness", self.max_loudness)
        log.debug(self.sampled_dynamics[0])
        log.debug(self.sampled_dynamics[0] / self.max_loudness)
        log.debug((self.sampled_dynamics[0] / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base)

        self.processed_dynamics = [(dynamic / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base for dynamic in self.processed_dynamics]
        self.sampled_dynamics = [(dynamic / self.max_loudness) * (1 - self.dynamics_base) + self.dynamics_base for dynamic in self.sampled_dynamics]
        log.debug(self.sampled_dynamics[0])

        self.average_loudness /= len(self.sampled_smoothed_avg_loudness)

    def init_sample_features(self):
        pass  # TODO: Implement

    def add_features_scaled(self):
        pass  # TODO: Implement

    def divide_features(self):
        pass  # TODO: Implement

    def sample(self, feature_type, options):
        pass  # TODO: Implement



    

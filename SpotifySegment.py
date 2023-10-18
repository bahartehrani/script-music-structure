import AudioUtil as audioUtil
try_remove_percussion = False
equalize_bass = False

class SpotifySegment:
    def __init__(self, segment):
        self.segment = segment
        self.start = segment['start']
        self.duration = segment['duration']
        self.loudness_start = segment['loudness_start']
        self.loudness_max_time = segment['loudness_max_time']
        self.loudness_max = segment['loudness_max']
        self.loudness_end = segment['loudness_end']

        self.pitches = []
        self.noise = []

        self.cluster = -1
        self.tsne_coord = [0, 0]

        self.tonality_angle = 0  # [0,1] Angle of vector on circle of fifths
        self.tonality_radius = 0  # [0,1] Radius of vector on circle of fifths
        self.tonality_energy = 0  # [0,1] Total energy of all the pitches
        self.percussiony = 0  # [0,1] segments larger than .15

        self.processed_pitch = False
        self.processed_pitch_smooth = False

        self.process_pitch()
        self.process_pitch_equalize_bass()

    def process_pitch(self):
        if self.processed_pitch:
            return
        if self.processed_pitch_smooth:
            raise ValueError("Processed pitchSmooth before setting initial pitch")

        for pitch in self.segment['pitches']:
            self.pitches.append(pitch)  # Assuming audioUtil.logCompression(pitch, GAMMA) is not required

        self.tonality_angle, self.tonality_radius, self.tonality_energy = audioUtil.tonality(self.pitches)

        min_duration = 0.2
        decay = 0.5
        shortness = 1 if self.duration < min_duration else decay - (self.duration - 0.15)

        for p in range(12):
            self.noise.append(max(0, self.pitches[(p + 11) % 12] + self.pitches[p] + self.pitches[(p + 1) % 12] - 2))

        self.percussiony = max(min(1, (1 - self.tonality_radius) * self.tonality_energy * 2) * shortness, 0)
        self.percussiony = max(0, min(1, 1 - self.tonality_radius * 6))
        self.percussiony = max(0, min(1, (1 - self.tonality_radius) * self.tonality_energy))
        self.percussiony = sum(self.noise) / 12 * 2

        self.processed_pitch = True

    def process_pitch_smooth(self, prev_segment, next_segment):
        if not try_remove_percussion:
            return
        if self.processed_pitch_smooth:
            return
        if not self.processed_pitch:
            raise ValueError("Processed pitchSmooth called before setting initial pitch")

        for p in range(len(self.pitches)):
            self.pitches[p] = (
                (1 - self.percussiony) * self.pitches[p] +
                self.percussiony *
                (prev_segment.pitches[p] * self.pitches[p] + next_segment.pitches[p] * self.pitches[p]) / 2
            )

        self.processed_pitch_smooth = True

    def process_pitch_equalize_bass(self):
        if equalize_bass:
            max_pitch = max(self.pitches)
            second_max_pitch = sorted([pitch for pitch in self.pitches if pitch != max_pitch])[-1]
            
            equalize_amount = 0
            scale = 1 / (1 - (max_pitch - second_max_pitch) * equalize_amount)
            for p in range(len(self.pitches)):
                self.pitches[p] = min(1, self.pitches[p] * scale)

    def get_pitches(self):
        return self.pitches

    def get_duration(self):
        return self.duration

    def get_start(self):
        return self.start

    def get_end(self):
        return self.start + self.duration

    def get_features(self):
        return self.pitches

    def set_cluster(self, i):
        self.cluster = i

    def set_tsne_coord(self, coord):
        self.tsne_coord = coord

    def get_tonal_energy(self):
        return self.tonality_energy

    def get_tonal_angle(self):
        return self.tonality_angle

    def get_loudness_features(self):
        return [
            audioUtil.loudness_perceived(self.loudness_start),
            audioUtil.loudness_perceived(self.loudness_max),
            self.loudness_max_time,
            audioUtil.loudness_perceived(self.loudness_end)
        ]

    def get_average_loudness(self, next_segment_loudness):
        loudness_features = self.get_loudness_features()
        start = loudness_features[0]
        end = next_segment_loudness
        max_val = loudness_features[1]
        max_time = loudness_features[2]

        avg_first = (max_val + start) / 2
        avg_second = (max_val + end) / 2
        avg = avg_first * max_time + avg_second * (1 - max_time)
        return avg
    
    def get_feature_by_name(self, feature):
        if feature == 'pitches':
            return self.get_pitches()
        if feature == 'loudness':
            return self.get_loudness_features()
        if feature == 'duration':
            return self.get_duration()
        return None


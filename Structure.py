from copy import deepcopy
from Section import Section
import MDS as MDS
import PathExtraction as PathExtraction
import math
import SSM as SSM
from collections import deque

def create_fixed_duration_structure_segments(sample_amount, sample_duration, duration):
    structure_segments = []
    start = 0
    while start + duration < sample_amount:
        structure_segments.append(
            Section({
                'start': start * sample_duration,
                'end': (start + duration) * sample_duration
            })
        )
        start += duration
    return structure_segments

def mds_color_segments(segments, path_ssm, strategy="overlap", coloring_strategy=None, color_both=False):
    if not segments:
        return segments

    distance_matrix = PathExtraction.get_distance_matrix(segments, path_ssm, strategy)
    segments_colored = mds_color_given_distance_matrix(segments, distance_matrix, coloring_strategy)

    if color_both:
        distance_matrix_categorical = PathExtraction.get_distance_matrix(segments, path_ssm, strategy, 0)
        segments_colored_categorical = mds_color_given_distance_matrix(
            segments, distance_matrix_categorical, coloring_strategy
        )

        for i in range(len(segments_colored)):
            segments_colored[i].cat_color_angle = segments_colored_categorical[i].color_angle
            segments_colored[i].cat_color_radius = segments_colored_categorical[i].color_radius
            segments_colored[i].cat_mds_feature = segments_colored_categorical[i].mds_feature

    return segments_colored

def mds_color_given_distance_matrix(segments, distance_matrix, coloring_strategy):
    colored_segments = []
    
    mds_coordinates = MDS.get_mds_coordinates(distance_matrix, coloring_strategy)
    mds_feature = MDS.get_mds_feature(distance_matrix)
    
    for index, segment in enumerate(segments):
        angle, radius = MDS.get_angle_and_radius(mds_coordinates[index])
        new_segment = segment.clone()
        new_segment.color_angle = angle
        new_segment.color_radius = radius
        new_segment.mds_feature = mds_feature[index]
        colored_segments.append(new_segment)
    
    # sort from small to high
    colored_segments.sort(key=lambda x: x.color_angle)

    largest_gap = 1 - colored_segments[-1].color_angle + colored_segments[0].color_angle
    largest_gap_angle = colored_segments[0].color_angle
    for i in range(1, len(colored_segments)):
        gap = colored_segments[i].color_angle - colored_segments[i - 1].color_angle
        if gap > largest_gap:
            largest_gap = gap
            largest_gap_angle = colored_segments[i].color_angle

    for segment in colored_segments:
        segment.color_angle = (1 + (segment.color_angle - largest_gap_angle)) % 1

    # sort by family and start time
    colored_segments.sort(key=lambda x: (x.group_id, x.start))
    
    return colored_segments

allow_overlap = True
wiggle = True
wiggle_size = 3
def find_mute_decomposition(path_ssm, structure_segments, sample_duration, strategy="classic", mute_type="or", 
                            update_callback=None, comparison_property="fitness", min_duration_seconds=2, 
                            min_fitness=0.02):
    
    track_end = structure_segments[-1].end
    structure_sections = []
    sorted_structure_sections = []
    segments = []

    separate_segment_sets = [structure_segments]
    ssm = path_ssm
    i = 0
    max_repeats = 14
    
    while len(separate_segment_sets) > 0 and i < max_repeats:
        separate_candidate_sets = compute_separate_structure_candidates(
            ssm, separate_segment_sets, strategy, min_duration_seconds)
        all_candidates = [item for sublist in separate_candidate_sets for item in sublist]
        
        if not all_candidates:
            break
        
        all_candidates_sorted = sorted(all_candidates, key=lambda x: getattr(x, comparison_property, None) , reverse=True)
        all_candidates_d = deque(all_candidates_sorted)
        best = all_candidates_d.popleft()
        initial_fitness = best.fitness

        if wiggle: 
            best = find_better_fit(ssm, best, wiggle_size, comparison_property, strategy, sorted_structure_sections, 
                                   min_duration_seconds)

        if best is None or getattr(best, comparison_property, None) <= min_fitness or math.isnan(getattr(best, comparison_property, None)):
            break
        
        group_id = i
        best.group_id = group_id

        path_family = get_path_family(best, sample_duration, group_id)
        extended_path_family = get_extended_path_family(best, sample_duration, group_id, path_ssm, strategy)
        non_overlapping_extended_path_family = add_non_overlapping(path_family, extended_path_family)
        pruned_paths = prune_low_confidence(non_overlapping_extended_path_family, 0.1)

        for path in pruned_paths:
            path.normalized_score = best.normalized_score 
            path.normalized_coverage = best.normalized_coverage
            path.fitness = best.fitness
            path.init_fitness = initial_fitness

        sorted_structure_sections.extend(pruned_paths)
        structure_sections.extend(pruned_paths)

        pruned_paths_in_samples = []
        for section in pruned_paths:
            clone = section.copy()
            clone.start = int(clone.start / sample_duration)
            clone.end = int(clone.end / sample_duration)
            pruned_paths_in_samples.append(clone)

        ssm = SSM.mute_or(ssm, pruned_paths_in_samples)

        sorted_structure_sections.sort(key=lambda x: x.start)

        if update_callback:
            update_callback(structure_sections)

        separate_segment_sets = subtract_structure_from_segments(
            deepcopy(separate_segment_sets), sorted_structure_sections, track_end, min_duration_seconds)

        all_segments = [item for sublist in separate_segment_sets for item in sublist]
        for segment in all_segments:
            segment.group_id = i
        segments.extend(all_segments)
        
        i += 1

    return sorted_structure_sections, i, segments

def compute_separate_structure_candidates(path_ssm, separate_segment_sets, strategy, min_duration_seconds=1, max_ratio=0.25):
    separate_candidate_sets = []
    
    for segments in separate_segment_sets:
        candidates = compute_structure_candidates(path_ssm, segments, min_duration_seconds, max_ratio, strategy)
        separate_candidate_sets.append(candidates)
    
    return separate_candidate_sets

def compute_structure_candidates(path_ssm, structure_segments, min_duration_seconds=1, max_ratio=0.4, strategy=None):
    sample_amount = path_ssm.height
    segment_amount = len(structure_segments)
    sample_duration = path_ssm.sample_duration
    max_length = max_ratio * sample_amount * sample_duration
    score_matrix_buffer = PathExtraction.create_score_matrix_buffer(sample_amount)
    candidates = []
    print('tbahar')
    print(segment_amount)

    for start in range(segment_amount):
        for end in range(start, segment_amount):
            start_in_seconds = structure_segments[start].start
            end_in_seconds = structure_segments[end].end
            segment_length_in_seconds = end_in_seconds - start_in_seconds

            if segment_length_in_seconds < min_duration_seconds or segment_length_in_seconds > max_length:
                continue

            start_in_samples = math.floor(start_in_seconds / sample_duration)
            end_in_samples = math.floor(end_in_seconds / sample_duration)

            candidates.append(
                create_candidate(path_ssm, start_in_samples, end_in_samples, score_matrix_buffer, strategy, len(candidates))
            )

    return candidates

def create_candidate(path_ssm, start, end, score_matrix_buffer, strategy, group_id):
    segment_path_family_info = PathExtraction.compute_segment_path_family_info(
        path_ssm, start, end, score_matrix_buffer, strategy
    )
    
    start_in_seconds = start * path_ssm.sample_duration
    end_in_seconds = end * path_ssm.sample_duration

    candidate = Section({
                'start': start_in_seconds,
                'end': end_in_seconds,
                'group_id': group_id
            })
    candidate.path_family = segment_path_family_info.get('path_family')
    candidate.path_family_scores = segment_path_family_info.get('path_scores')
    candidate.score = segment_path_family_info.get('score')
    candidate.normalized_score = segment_path_family_info.get('normalized_score')
    candidate.coverage = segment_path_family_info.get('coverage')
    candidate.normalized_coverage = segment_path_family_info.get('normalized_coverage')
    candidate.fitness = segment_path_family_info.get('fitness')

    return candidate

def find_better_fit(
    path_ssm, 
    section, 
    sample_offset=4, 
    comparison_property=None, 
    strategy=None, 
    structure_sections=[], 
    min_duration_seconds=None
):
    sample_amount = path_ssm.height
    sample_duration = path_ssm.sample_duration

    start_in_samples = int(section.start / sample_duration)
    end_in_samples = int(section.end / sample_duration)

    score_matrix_buffer = PathExtraction.create_score_matrix_buffer(sample_amount)
    max_value = getattr(section, comparison_property, None) 
    best_fit = section

    for start_offset in range(-sample_offset, sample_offset):
        for end_offset in range(-sample_offset, sample_offset):
            start = start_in_samples + start_offset
            end = end_in_samples + end_offset

            start_in_seconds = start * sample_duration
            end_in_seconds = end * sample_duration

            if (
                start >= 0 
                and end < sample_amount 
                and end_in_seconds - start_in_seconds >= min_duration_seconds 
                and (allow_overlap or not overlap_with_structure_sections({"start": start_in_seconds, "end": end_in_seconds}, structure_sections))
            ):

                segment_path_family_info = PathExtraction.compute_segment_path_family_info(
                    path_ssm,
                    start,
                    end,
                    score_matrix_buffer,
                    strategy
                )

                segment_path_family_info[start] = start * sample_duration
                segment_path_family_info[end] = end * sample_duration

                if segment_path_family_info[comparison_property] > max_value:
                    # TODO: log how the max moves (maybe it needs more offset to reach optimum)
                    best_fit = segment_path_family_info
                    max_value = segment_path_family_info[comparison_property]

    return best_fit

def overlap_with_structure_sections(section, structure_sections):
    return any(section.overlaps(structure_section) for structure_section in structure_sections)

def get_path_family(section, sample_duration, group_id):
    path_family_sections = []

    if not section.path_family:
        return []

    for index, path in enumerate(section.path_family):
        start = path[-1][1] * sample_duration
        end = (path[0][1] + 1) * sample_duration
        path_confidence = section.path_family_scores[index] if section.path_family_scores else section.path_scores[index]
        
        new_section = Section({
            'start': start, 
            'end': end, 
            'confidence': path_confidence
        })
        new_section.path_family = section.path_family
        path_family_sections.append(new_section)

    return path_family_sections

def get_extended_path_family(section, sample_duration, group_id, path_ssm, strategy):
    score_matrix_buffer = PathExtraction.create_score_matrix_buffer(path_ssm.height)
    path_family_sections = []

    for index, path in enumerate(section.path_family):
        start = path[-1][1] * sample_duration
        end = (path[0][1] + 1) * sample_duration
        path_confidence = section.path_family_scores[index] if section.path_family_scores else section.path_scores[index]

        new_section = Section({
            'start': start, 
            'end': end, 
            'group_id': group_id,
            'confidence': path_confidence
        })
        new_section.path_family = section.path_family
        path_family_sections.append(new_section)

    for family_section in path_family_sections:
        start_in_samples = math.floor(family_section.start / sample_duration)
        end_in_samples = math.floor(family_section.end / sample_duration)
        # print('tbahar')
        # print(start_in_samples)
        # print(end_in_samples)

        path_family_info = PathExtraction.compute_segment_path_family_info(
            path_ssm,
            start_in_samples,
            end_in_samples,
            score_matrix_buffer,
            strategy
        )

        for index, path in enumerate(path_family_info['path_family']):
            start = path[-1][1] * sample_duration
            end = (path[0][1] + 1) * sample_duration

            new_section = Section({
                'start': start, 
                'end': end, 
                'group_id': group_id,
                'confidence': family_section.confidence * path_family_info['path_scores'][index]
            })
            new_section.path_family = section.path_family
            path_family_sections.append(new_section)

    return path_family_sections

def add_non_overlapping(sections_a, sections_b, overlap_ratio_threshold=0):
    all_sections = sections_a.copy()

    sorted_sections_b = sorted(sections_b, key=lambda x: x.confidence)

    for section_b in sorted_sections_b:
        most_overlap_ratio = 0
        for section_a in all_sections:
            if section_a.overlaps(section_b):
                duration_a = section_a.end - section_a.start
                duration_b = section_b.end - section_b.start
                overlap = compute_overlap_size(section_a, section_b)
                overlap_ratio = overlap / (duration_a + duration_b)
                if overlap_ratio > most_overlap_ratio:
                    most_overlap_ratio = overlap_ratio
        if 0 < most_overlap_ratio < overlap_ratio_threshold:
            all_sections.append(section_b)
        elif most_overlap_ratio <= 0:
            all_sections.append(section_b)

    return all_sections

def compute_overlap_size(a, b):
    if a.start <= b.start and a.end > b.start:
        return a.end - b.start
    if b.start <= a.start and b.end > a.start:
        return b.end - a.start
    if a.start < b.start and a.end >= b.end:
        return b.end - b.start
    if b.start < a.start and b.end >= a.end:
        return a.end - a.start
    if a.start >= b.start and a.end <= b.end:
        return a.end - a.start
    if b.start >= a.start and b.end <= a.end:
        return b.end - b.start
    return 0

def prune_low_confidence(sections, min_confidence):
    pruned_sections = [section for section in sections if section.confidence >= min_confidence]
    return pruned_sections

def subtract_structure_from_segments(separate_segment_sets, sorted_structure_sections, track_end, smallest_allowed_size):
    new_separate_segment_sets = []
    all_segments = [segment for subset in separate_segment_sets for segment in subset]

    free_ranges = get_free_ranges(sorted_structure_sections, track_end, smallest_allowed_size)
    for range_ in free_ranges:
        range_start, range_end = range_
        segment_set = []

        previous_border = range_start
        for segment in all_segments:
            if previous_border < segment.start < range_end:
                segment_set.append(Section({
                'start': previous_border,
                'end': segment.start
            }))
                previous_border = segment.start
            segment_set.append(Section({
                'start': previous_border,
                'end': range_end
            }))
        new_separate_segment_sets.append(segment_set)

    return new_separate_segment_sets

def get_free_ranges(structure_sections, track_end, smallest_allowed_size):
    free_ranges = [[0, track_end]]
    
    for section in structure_sections:
        new_ranges = []
        i = 0
        while i < len(free_ranges):
            range_ = free_ranges[i]
            range_start, range_end = range_
            
            # If range is encapsulated by section, remove
            if range_start >= section.start and range_end <= section.end:
                i += 1
                continue
            # If range overlaps section on left, cut range short, clip end
            elif range_start <= section.start and range_end > section.start and range_end <= section.end:
                range_[1] = section.start
                new_ranges.append(range_)
                i += 1
            # If range overlaps section on right, cut range short, clip start
            elif range_start < section.end and range_end >= section.start and range_start >= section.start:
                range_[0] = section.end
                new_ranges.append(range_)
                i += 1
            # If range encapsulates section, cut range in half
            elif range_start < section.start and range_end > section.end:
                new_ranges.append([range_start, section.start])
                new_ranges.append([section.end, range_end])
                i += 1
            else:
                new_ranges.append(range_)
                i += 1
        free_ranges = new_ranges

    return [range_ for range_ in free_ranges if range_[1] - range_[0] >= smallest_allowed_size]

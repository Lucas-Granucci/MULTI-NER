def get_entities(seq):
    """
    Gets entities from sequence in BIO format.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):  # Adding 'O' to handle end of the sequence
        if chunk == 'O':
            tag, type_ = 'O', ''
        else:
            tag, type_ = chunk.split('-')

        # Check for end of chunk
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))

        # Check for start of new chunk
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i

        prev_tag, prev_type = tag, type_

    return chunks

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Checks if a new chunk starts between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        bool: True if a new chunk starts.
    """
    if tag == 'B' or (tag == 'I' and prev_type != type_):
        return True
    return False

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        bool: True if the chunk ended.
    """
    if prev_tag == 'B' and (tag == 'B' or tag == 'O' or prev_type != type_):
        return True
    if prev_tag == 'I' and (tag == 'B' or tag == 'O' or prev_type != type_):
        return True
    return False


def f1_score(y_true, y_pred, config, detailed_breakdown=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    avg_score = {'p': p, 'r': r, 'f1': score}
    if not detailed_breakdown:
        return avg_score
    else:
        label_score = {}
        for label in config["data"]["labels"]:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            label_score[label] = {'p': p_label, 'r': r_label, 'f1': score_label}
        return label_score, avg_score
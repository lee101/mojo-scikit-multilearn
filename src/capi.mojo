"""Dense multi-label kernels exported through a stable C ABI."""

from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime IW = simd_width_of[DType.int64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def squared_distance(a: FPtr, b: FPtr, size: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= size:
        var delta = a.load[width=W](i) - b.load[width=W](i)
        acc += delta * delta
        i += W
    var total = acc.reduce_add()
    while i < size:
        var delta = a[i] - b[i]
        total += delta * delta
        i += 1
    return total


def knn_query_row(
    train: FPtr,
    query: FPtr,
    indices: IPtr,
    distances: FPtr,
    train_rows: Int,
    features: Int,
    neighbors: Int,
    exclude_self: Bool,
    query_offset: Int,
    query_row: Int,
):
    var base = query_row * neighbors
    for slot in range(neighbors):
        indices[base + slot] = -1
        distances[base + slot] = 1.7976931348623157e308

    for train_row in range(train_rows):
        if exclude_self and train_row == query_offset + query_row:
            continue
        var distance = squared_distance(
            query + query_row * features,
            train + train_row * features,
            features,
        )
        if distance >= distances[base + neighbors - 1]:
            continue

        var slot = neighbors - 1
        while slot > 0 and distances[base + slot - 1] > distance:
            distances[base + slot] = distances[base + slot - 1]
            indices[base + slot] = indices[base + slot - 1]
            slot -= 1
        distances[base + slot] = distance
        indices[base + slot] = Int64(train_row)


def knn_query(
    train: FPtr,
    query: FPtr,
    indices: IPtr,
    distances: FPtr,
    train_rows: Int,
    features: Int,
    query_rows: Int,
    neighbors: Int,
    exclude_self: Bool,
    query_offset: Int,
):
    for query_row in range(query_rows):
        knn_query_row(
            train,
            query,
            indices,
            distances,
            train_rows,
            features,
            neighbors,
            exclude_self,
            query_offset,
            query_row,
        )


def neighbor_label_counts(
    indices: IPtr,
    labels: IPtr,
    counts: IPtr,
    query_rows: Int,
    neighbors: Int,
    label_count: Int,
):
    for query_row in range(query_rows):
        var count_base = query_row * label_count
        var label = 0
        var zeros = SIMD[DType.int64, IW](0)
        while label + IW <= label_count:
            counts.store(count_base + label, zeros)
            label += IW
        while label < label_count:
            counts[count_base + label] = 0
            label += 1

        for slot in range(neighbors):
            var train_row = Int(indices[query_row * neighbors + slot])
            var label_base = train_row * label_count
            label = 0
            while label + IW <= label_count:
                var sums = counts.load[width=IW](count_base + label)
                sums += labels.load[width=IW](label_base + label)
                counts.store(count_base + label, sums)
                label += IW
            while label < label_count:
                counts[count_base + label] += labels[label_base + label]
                label += 1


def take_labelsets(
    combinations: IPtr,
    assignments: IPtr,
    result: IPtr,
    rows: Int,
    label_count: Int,
):
    for row in range(rows):
        var combination = Int(assignments[row])
        for label in range(label_count):
            result[row * label_count + label] = combinations[
                combination * label_count + label
            ]


@export("msml_knn_query")
def msml_knn_query(
    train_addr: Int,
    query_addr: Int,
    indices_addr: Int,
    distances_addr: Int,
    train_rows: Int,
    features: Int,
    query_rows: Int,
    neighbors: Int,
    exclude_self: Int,
    query_offset: Int,
) abi("C"):
    knn_query(
        FPtr(unsafe_from_address=train_addr),
        FPtr(unsafe_from_address=query_addr),
        IPtr(unsafe_from_address=indices_addr),
        FPtr(unsafe_from_address=distances_addr),
        train_rows,
        features,
        query_rows,
        neighbors,
        Bool(exclude_self),
        query_offset,
    )


@export("msml_neighbor_label_counts")
def msml_neighbor_label_counts(
    indices_addr: Int,
    labels_addr: Int,
    counts_addr: Int,
    query_rows: Int,
    neighbors: Int,
    label_count: Int,
) abi("C"):
    neighbor_label_counts(
        IPtr(unsafe_from_address=indices_addr),
        IPtr(unsafe_from_address=labels_addr),
        IPtr(unsafe_from_address=counts_addr),
        query_rows,
        neighbors,
        label_count,
    )


@export("msml_take_labelsets")
def msml_take_labelsets(
    combinations_addr: Int,
    assignments_addr: Int,
    result_addr: Int,
    rows: Int,
    label_count: Int,
) abi("C"):
    take_labelsets(
        IPtr(unsafe_from_address=combinations_addr),
        IPtr(unsafe_from_address=assignments_addr),
        IPtr(unsafe_from_address=result_addr),
        rows,
        label_count,
    )

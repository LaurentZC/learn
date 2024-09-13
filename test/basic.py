from math import sqrt

def merge(lst, left, mid, right):
    left_lst = lst[left:mid + 1]
    right_lst = lst[mid + 1:right + 1]

    left_idx = right_idx = 0
    sorted_idx = left

    while left_idx < len(left_lst) and right_idx < len(right_lst):
        if left_lst[left_idx] <= right_lst[right_idx]:
            lst[sorted_idx] = left_lst[left_idx]
            left_idx += 1
        else:
            lst[sorted_idx] = right_lst[right_idx]
            right_idx += 1
        sorted_idx += 1

    while left_idx < len(left_lst):
        lst[sorted_idx] = left_lst[left_idx]
        left_idx += 1
        sorted_idx += 1

    while right_idx < len(right_lst):
        lst[sorted_idx] = right_lst[right_idx]
        right_idx += 1
        sorted_idx += 1


def merge_sort(array, left_index, right_index):
    if left_index < right_index:
        mid_index = (left_index + right_index) // 2
        merge_sort(array, left_index, mid_index)
        merge_sort(array, mid_index + 1, right_index)
        merge(array, left_index, mid_index, right_index)


if __name__ == "__main__":
    sample_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", sample_array)
    merge_sort(sample_array, 0, len(sample_array) - 1)
    print("Sorted array:", sample_array)

    sample_array.append(70)
    sample_array.insert(1, 80)

    print("Array after insert:", sample_array)
    merge_sort(sample_array, 0, len(sample_array) - 1)
    print("Sorted array:", sample_array)

    sqrt_array = [sqrt(x) for x in sample_array[1:4]]
    print(f"Array after sqrt: {[f'{x:.3f}' for x in sqrt_array]}")
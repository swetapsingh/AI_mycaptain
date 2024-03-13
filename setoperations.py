def get_sets_from_user():
    sets = []
    num_sets = int(input("Enter the number of sets: "))
    for i in range(num_sets):
        elements = input("Enter the elements of set {}: ".format(i + 1)).split(',')
        set_elements = {int(element) for element in elements}
        sets.append(set_elements)
    return sets

def set_operations(sets):
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            set1 = sets[i]
            set2 = sets[j]
            union_set = set1.union(set2)
            intersection_set = set1.intersection(set2)
            difference_set1 = set1.difference(set2)
            difference_set2 = set2.difference(set1)
            symmetric_difference_set = set1.symmetric_difference(set2)

            print("Union of set {} and set {} is {}".format(set1, set2, union_set))
            print("Intersection of set {} and set {} is {}".format(set1, set2, intersection_set))
            print("Difference of set {} and set {} is {}".format(set1, set2, difference_set1))
            print("Difference of set {} and set {} is {}".format(set2, set1, difference_set2))
            print("Symmetric difference of set {} and set {} is {}".format(set1, set2, symmetric_difference_set))


# Get sets from user
sets = get_sets_from_user()

# Perform set operations
set_operations(sets)

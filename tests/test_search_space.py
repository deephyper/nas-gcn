from nas_gcn.esol.problem import Problem as ESOL_Problem


print(ESOL_Problem)

space = ESOL_Problem.build_search_space()
print("#Operations: ", space.num_nodes)

# arch_seq = [0 for _ in range(space.num_nodes)]
arch_seq = [17257, 0, 11863, 1, 0, 17851, 1, 1, 0, 7]

model = ESOL_Problem.get_keras_model(arch_seq)

model.summary()
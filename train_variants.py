from train_embeddings import train

for manifold in ['poincare', 'euclidean']:
    for dim in [2]:
        train(
            variant='%s-%s-shared-root' % (manifold, dim),
            manifold=manifold,
            dim=dim,
            lr=1.0,
            batch_size=50,
            epochs=10
        )

/*
 * this is taken from R version of igraph; sadly not available in the C version
 * of the library.
 */

int bundle_igraph_layout_kamada_kawai(const igraph_t *graph, igraph_matrix_t *res,
        igraph_integer_t maxiter,
        igraph_real_t epsilon, igraph_real_t kkconst, 
        const igraph_vector_t *weights,
        const igraph_vector_t *minx, const igraph_vector_t *maxx,
        const igraph_vector_t *miny, const igraph_vector_t *maxy) {

    igraph_integer_t no_nodes=igraph_vcount(graph);
    igraph_integer_t no_edges=igraph_ecount(graph);
    igraph_real_t L, L0=sqrt(no_nodes);  
    igraph_matrix_t dij, lij, kij;
    igraph_real_t max_dij;
    igraph_vector_t D1, D2;
    igraph_integer_t i, j, m;

    if (maxiter < 0) {
        IGRAPH_ERROR("Number of iterations must be non-negatice in "
                "Kamada-Kawai layout", IGRAPH_EINVAL);
    }
    if (kkconst <= 0) {
        IGRAPH_ERROR("`K' constant must be positive in Kamada-Kawai layout",
                IGRAPH_EINVAL);
    }

    if ((igraph_matrix_nrow(res) != no_nodes ||
                igraph_matrix_ncol(res) != 2)) {
        IGRAPH_ERROR("Invalid start position matrix size in "
                "Kamada-Kawai layout", IGRAPH_EINVAL);
    }
    if (weights && igraph_vector_size(weights) != no_edges) {
        IGRAPH_ERROR("Invalid weight vector length", IGRAPH_EINVAL);
    }

    if (minx && igraph_vector_size(minx) != no_nodes) {
        IGRAPH_ERROR("Invalid minx vector length", IGRAPH_EINVAL);
    }
    if (maxx && igraph_vector_size(maxx) != no_nodes) {
        IGRAPH_ERROR("Invalid maxx vector length", IGRAPH_EINVAL);
    }
    if (minx && maxx && !igraph_vector_all_le(minx, maxx)) {
        IGRAPH_ERROR("minx must not be greater than maxx", IGRAPH_EINVAL);
    }
    if (miny && igraph_vector_size(miny) != no_nodes) {
        IGRAPH_ERROR("Invalid miny vector length", IGRAPH_EINVAL);
    }
    if (maxy && igraph_vector_size(maxy) != no_nodes) {
        IGRAPH_ERROR("Invalid maxy vector length", IGRAPH_EINVAL);
    }
    if (miny && maxy && !igraph_vector_all_le(miny, maxy)) {
        IGRAPH_ERROR("miny must not be greater than maxy", IGRAPH_EINVAL);
    }

    if (no_nodes <= 1) { return 0; }

    IGRAPH_MATRIX_INIT_FINALLY(&dij, no_nodes, no_nodes);
    IGRAPH_MATRIX_INIT_FINALLY(&kij, no_nodes, no_nodes);
    IGRAPH_MATRIX_INIT_FINALLY(&lij, no_nodes, no_nodes);

    if (weights && igraph_vector_min(weights) < 0) {
        IGRAPH_CHECK(igraph_shortest_paths_bellman_ford(graph, &dij, igraph_vss_all(),
                    igraph_vss_all(), weights,
                    IGRAPH_ALL));
    } else {

        IGRAPH_CHECK(igraph_shortest_paths_dijkstra(graph, &dij, igraph_vss_all(),
                    igraph_vss_all(), weights,
                    IGRAPH_ALL));
    }

    max_dij = 0.0;
    for (i=0; i<no_nodes; i++) {
        for (j=i+1; j<no_nodes; j++) {
            if (!igraph_finite(MATRIX(dij, i, j))) { continue; }
            if (MATRIX(dij, i, j) > max_dij) { max_dij = MATRIX(dij, i, j); }
        }
    }
    for (i=0; i<no_nodes; i++) {
        for (j=0; j<no_nodes; j++) {
            if (MATRIX(dij, i, j) > max_dij) { MATRIX(dij, i, j) = max_dij; }
        }
    }

    L = L0 / max_dij;
    for (i=0; i<no_nodes; i++) {
        for (j=0; j<no_nodes; j++) {
            igraph_real_t tmp=MATRIX(dij, i, j) * MATRIX(dij, i, j);
            if (i==j) { continue; }
            MATRIX(kij, i, j) = kkconst / tmp;
            MATRIX(lij, i, j) = L * MATRIX(dij, i, j);
        }
    }

    /* Initialize delta */
    IGRAPH_VECTOR_INIT_FINALLY(&D1, no_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&D2, no_nodes);
    for (m=0; m<no_nodes; m++) {
        igraph_real_t myD1=0.0, myD2=0.0;
        for (i=0; i<no_nodes; i++) { 
            igraph_real_t dx, dy, mi_dist;
            if (i==m) { continue; }
            dx=MATRIX(*res, m, 0) - MATRIX(*res, i, 0);
            dy=MATRIX(*res, m, 1) - MATRIX(*res, i, 1);
            mi_dist=sqrt(dx * dx + dy * dy);
            myD1 += MATRIX(kij, m, i) * (dx - MATRIX(lij, m, i) * dx / mi_dist);
            myD2 += MATRIX(kij, m, i) * (dy - MATRIX(lij, m, i) * dy / mi_dist);
        }
        VECTOR(D1)[m] = myD1;
        VECTOR(D2)[m] = myD2;
    }

    for (j=0; j<maxiter; j++) {
        igraph_real_t myD1, myD2, A, B, C;
        igraph_real_t max_delta, delta_x, delta_y;
        igraph_real_t old_x, old_y, new_x, new_y;

        myD1=0.0, myD2=0.0, A=0.0, B=0.0, C=0.0;

        /* Select maximal delta */
        m=0; max_delta=-1;
        for (i=0; i<no_nodes; i++) {
            igraph_real_t delta=(VECTOR(D1)[i] * VECTOR(D1)[i] + 
                    VECTOR(D2)[i] * VECTOR(D2)[i]);
            if (delta > max_delta) { 
                m=i; max_delta=delta;
            }
        }

        if (max_delta < epsilon) { break; }
        old_x=MATRIX(*res, m, 0);
        old_y=MATRIX(*res, m, 1);

        /* Calculate D1 and D2, A, B, C */
        for (i=0; i<no_nodes; i++) {
            igraph_real_t dx, dy, dist, den;
            if (i==m) { continue; }
            dx=old_x - MATRIX(*res, i, 0);
            dy=old_y - MATRIX(*res, i, 1);
            dist=sqrt(dx * dx + dy * dy);
            den=dist * (dx * dx + dy * dy);
            A += MATRIX(kij, m, i) * (1 - MATRIX(lij, m, i) * dy * dy / den);
            B += MATRIX(kij, m, i) * MATRIX(lij, m, i) * dx * dy / den;
            C += MATRIX(kij, m, i) * (1 - MATRIX(lij, m, i) * dx * dx / den);
        }
        myD1 = VECTOR(D1)[m];
        myD2 = VECTOR(D2)[m];

        /* Need to solve some linear equations */
        delta_y = (B * myD1 - myD2 * A) / (C * A - B * B);
        delta_x = - (myD1 + B * delta_y) / A;

        new_x = old_x + delta_x;
        new_y = old_y + delta_y;

        /* Limits, if given */
        if (minx && new_x < VECTOR(*minx)[m]) { new_x = VECTOR(*minx)[m]; }
        if (maxx && new_x > VECTOR(*maxx)[m]) { new_x = VECTOR(*maxx)[m]; }
        if (miny && new_y < VECTOR(*miny)[m]) { new_y = VECTOR(*miny)[m]; }
        if (maxy && new_y > VECTOR(*maxy)[m]) { new_y = VECTOR(*maxy)[m]; }

        /* Update delta, only with/for the affected node */
        VECTOR(D1)[m] = VECTOR(D2)[m] = 0.0;
        for (i=0; i<no_nodes; i++) {
            igraph_real_t old_dx, old_dy, old_mi, new_dx, new_dy, new_mi_dist, old_mi_dist;
            if (i==m) { continue; }
            old_dx=old_x - MATRIX(*res, i, 0);
            old_dy=old_y - MATRIX(*res, i, 1);
            old_mi_dist=sqrt(old_dx * old_dx + old_dy * old_dy);
            new_dx=new_x - MATRIX(*res, i, 0);
            new_dy=new_y - MATRIX(*res, i, 1);
            new_mi_dist=sqrt(new_dx * new_dx + new_dy * new_dy);

            VECTOR(D1)[i] -= MATRIX(kij, m, i) * 
                (-old_dx + MATRIX(lij, m, i) * old_dx / old_mi_dist);
            VECTOR(D2)[i] -= MATRIX(kij, m, i) *
                (-old_dy + MATRIX(lij, m, i) * old_dy / old_mi_dist);
            VECTOR(D1)[i] += MATRIX(kij, m, i) *
                (-new_dx + MATRIX(lij, m, i) * new_dx / new_mi_dist);
            VECTOR(D2)[i] += MATRIX(kij, m, i) *
                (-new_dy + MATRIX(lij, m, i) * new_dy / new_mi_dist);

            VECTOR(D1)[m] += MATRIX(kij, m, i) *
                (new_dx - MATRIX(lij, m, i) * new_dx / new_mi_dist);
            VECTOR(D2)[m] += MATRIX(kij, m, i) *
                (new_dy - MATRIX(lij, m, i) * new_dy / new_mi_dist);
        }

        /* Update coordinates*/
        MATRIX(*res, m, 0) = new_x;
        MATRIX(*res, m, 1) = new_y;
    }

    igraph_vector_destroy(&D2);
    igraph_vector_destroy(&D1);
    igraph_matrix_destroy(&lij);
    igraph_matrix_destroy(&kij);
    igraph_matrix_destroy(&dij);
    IGRAPH_FINALLY_CLEAN(5);

    return 0;
}

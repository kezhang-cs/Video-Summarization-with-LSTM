import numpy
import theano
import theano.tensor as T
import time


def adam_opt(model, train_set, valid_set, model_save_dir,
                          minibatch=64, valid_period=1, total_period = 0,
                          disp_period = 1, n_iters=10000000, lr=0.001,
                          beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1-1e-8):
    """
    Adam optimizer (ICLR 2015)
    """
    # initialize learning rate
    lr_file = open(model_save_dir+'lr.txt', 'w')
    lr_file.write(str(lr))
    lr_file.close()
    lr = theano.shared(numpy.array(lr).astype(theano.config.floatX))

    updates = []
    all_grads = theano.grad(model.costs[0], model.params)
    i = theano.shared(numpy.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)
    lr_t = lr * (T.sqrt(fix2) / fix1)

    for p, g in zip(model.params, all_grads):
        m = theano.shared(
            numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m)
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    grad_and_cost = all_grads
    grad_and_cost.append(model.costs[0])
    train_grad_f = theano.function(model.inputs, grad_and_cost, on_unused_input='warn')
    train_update_params_f = theano.function(grad_and_cost[0:-1], None, updates=updates)
    if valid_set != None:
        valid_f = theano.function(model.inputs, model.costs, on_unused_input='warn')

    # create log file
    log_file = open(model_save_dir + 'log.txt', 'w')
    log_file.write('adam_optimizer\n')
    log_file.write('lr=%f, beta1=%f, beta2=%f, epsilon=%f, gamma=%f\n' % (lr.get_value(), beta1, beta2, epsilon, gamma))
    log_file.close()

    print '... training with Adam optimizer'
    cap_count = 0
    train_cost = []
    t0 = time.clock()
    try:
        for u in xrange(n_iters):
            if u % 10 == 0:
                # refresh lr
                try:
                    lr_file = open(model_save_dir+'_lr.txt', 'r')
                    lr.set_value(float(lr_file.readline().rstrip()))
                    lr_file.close()
                except IOError:
                    pass

            grads = [numpy.zeros_like(p).astype(theano.config.floatX) for p in model.params]
            mb_cost = []
            for i in train_set.iterate(True):
                tmp = train_grad_f(*i)
                new_grads = tmp[0:-1]
                mb_cost.append(tmp[-1])
                grads = [g1+g2 for g1, g2 in zip(grads, new_grads)]
            grads = [g/numpy.array(minibatch) for g in grads]
            train_update_params_f(*grads)
            train_cost.append(numpy.mean(mb_cost))

            # output some information
            if u % disp_period == 0 and u > 0:
                p_now = numpy.concatenate([p.get_value().flatten() for p in model.params])
                if u < 4*disp_period:
                    p_last = numpy.zeros_like(p_now)
                    delta_last = numpy.zeros_like(p_now)
                delta_now = p_now - p_last
                angle = numpy.arccos(numpy.dot(delta_now, delta_last) / numpy.linalg.norm(delta_now) / numpy.linalg.norm(delta_last))
                angle = angle / numpy.pi * 180
                p_last = p_now
                delta_last = delta_now
                t1 = time.clock()
                print 'period=%d, update=%d, mb_cost=[%.4f], |delta|=[%.2e], angle=[%.1f], lr=[%.6f], t=[%.2f]sec' % \
                      (u/valid_period, u, numpy.mean(train_cost), numpy.mean(abs(delta_now[0:10000])), angle, lr.get_value(), (t1-t0))
                t0 = time.clock()
                train_cost = []

            if u % valid_period == 0 and u > 0:
                model.save_to_file(model_save_dir, total_period + (u)/valid_period)
                valid_loss = []
                valid_acc = []
                train_loss = []
                train_acc = []
                for i in valid_set.iterate(True):
                    loss, acc = valid_f(*i)
                    valid_loss.append(loss)
                    valid_acc.append(acc)
                for i in train_set.iterate(True):
                    loss, acc = valid_f(*i)
                    train_loss.append(loss)
                    train_acc.append(acc)
                cap_count += valid_period*minibatch
                output_info = 'period %i, valid loss=[%.4f], valid acc=[%.4f], train loss=[%.4f], train acc=[%.4f]' % \
                              (u/valid_period, numpy.mean(valid_loss), numpy.mean(valid_acc), numpy.mean(train_loss), numpy.mean(train_acc))
                print output_info
                log_file = open(model_save_dir + 'log.txt', 'a')
                log_file.write(output_info+'\n')
                log_file.close()
    except KeyboardInterrupt:
        print 'Training interrupted.'

# utils for inflation

def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                         inflated_param_names):
    """Inflate a conv module from 2d to 3d.

    Args:
        conv3d (nn.Module): The destination conv3d module.
        state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
        module_name_2d (str): The name of corresponding conv module in the
            2d model.
        inflated_param_names (list[str]): List of parameters that have been
            inflated.
    """
    weight_2d_name = module_name_2d + '.weight'

    conv2d_weight = state_dict_2d[weight_2d_name]
    kernel_t = conv3d.weight.data.shape[2]

    new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
        conv3d.weight) / kernel_t
    conv3d.weight.data.copy_(new_weight)
    inflated_param_names.append(weight_2d_name)

    if getattr(conv3d, 'bias') is not None:
        bias_2d_name = module_name_2d + '.bias'
        conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
        inflated_param_names.append(bias_2d_name)

def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                       inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                raise Warning(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')

            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

def ConvBlock_inflate(module, name, state_dict_r2d, inflated_param_names):
    # Residual part
    if module.res_conv:
        original_conv_name, original_bn_name = name + '.residual_conv', name + '.residual_bn'
        if original_conv_name + '.weight' not in state_dict_r2d:
            raise Warning(f'Module not exist in the state_dict_r2d'
                            f': {original_conv_name}')
        if original_bn_name + '.weight' not in state_dict_r2d:
            raise Warning(f'Module not exist in the state_dict_r2d'
                            f': {original_bn_name}')
        _inflate_conv_params(module.residual_conv, state_dict_r2d, original_conv_name, inflated_param_names)
        _inflate_bn_params(module.residual_bn, state_dict_r2d, original_bn_name, inflated_param_names)

    convs = ['conv1', 'conv2', 'conv3']
    for i, c in enumerate(convs):
        # layer{X}.conv{n}
        original_conv_name = name + '.{}'.format(c)
        # layer{X}.bn{n}
        original_bn_name = original_conv_name.replace(c, 'bn{}'.format(str(i+1)))
        if original_conv_name + '.weight' not in state_dict_r2d:
            raise Warning(f'Module not exist in the state_dict_r2d'
                            f': {original_conv_name}')
        else:
            shape_2d = state_dict_r2d[original_conv_name +'.weight'].shape
            if c == 'conv1':
                conv3d_module, bn3d_module = module.conv1, module.bn1
            elif c == 'conv2':
                conv3d_module, bn3d_module = module.conv2, module.bn2
            else:
                conv3d_module, bn3d_module = module.conv3, module.bn3
            shape_3d = conv3d_module.weight.data.shape
            if shape_2d != shape_3d[:2] + shape_3d[3:]:
                raise Warning(f'Weight shape mismatch for '
                            f': {original_conv_name} : '
                            f'3d weight shape: {shape_3d}; '
                            f'2d weight shape: {shape_2d}. ')
            else:
                _inflate_conv_params(conv3d_module, state_dict_r2d,
                                     original_conv_name,
                                     inflated_param_names)

        if original_bn_name + '.weight' not in state_dict_r2d:
            raise Warning(f'Module not exist in the state_dict_r2d'
                            f': {original_bn_name}')
        else:
            _inflate_bn_params(bn3d_module, state_dict_r2d,
                               original_bn_name,
                               inflated_param_names)
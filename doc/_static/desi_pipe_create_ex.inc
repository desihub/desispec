.. code-block:: console

    $> desi_pipe create \
    --root /project/projectdirs/desi \
    --db-postgres \
    --data ./desi_test/sim \
    --redux ./desi_test/redux \
    --prod svdc \
    --basis /project/projectdirs/desi/spectro/templates/basis_templates/v3.1 \
    --calib /project/projectdirs/desi/spectro/desi_spectro_calib/trunk

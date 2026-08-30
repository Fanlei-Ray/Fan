#include "arm_Calculate.h"
#include <math.h>

#define pi 3.14159265358979323846f

/*
 * 【DH参数表：机械臂的“DNA”】
 * 图中 DH 表的长度已换算为米：220mm=0.220m，141mm=0.141m，75mm=0.075m。
 * theta_offset 是安装零位偏置，实际关节角由调用者传入并与该偏置相加。
 * 
 * 改进型 DH (Modified DH) 参数说明：
 * alpha： 连杆 i-1 绕 X 轴旋转的角度
 * a：     连杆 i-1 沿 X 轴平移的距离 (杆长)
 * theta： 连杆 i 绕 Z 轴旋转的角度 (加上偏移量)
 * d：     连杆 i 沿 Z 轴平移的距离 (偏置)
 */
const Arm_DH_Param_t Arm_DH_Table[ARM_CALCULATE_JOINT_COUNT] = {
  { 0.0f,       0.000f,  0.0f,        0.000f }, /* J1: DM-8009P (底座旋转) */
  {-pi / 2.0f, 0.000f,  0.0f,        0.000f }, /* J2: DM-8009P (大臂俯仰) */
  {-pi / 2.0f, 0.000f, pi / 2.0f,   0.000f }, /* J3: DM-4340P (大臂旋转) */
  {-pi / 2.0f, 0.000f,-pi / 2.0f,   0.220f }, /* J4: DM-4040  (肘关节俯仰) */
  {-pi / 2.0f, 0.000f,  0.0f,        0.000f }, /* J5: DM-4310  (小臂旋转) */
  { pi / 2.0f, 0.000f,-pi / 2.0f,   0.141f }, /* J6: DM-4310  (腕关节俯仰) */
  { pi / 2.0f, 0.075f,-pi / 2.0f,   0.000f }  /* J7: DM-4310  (末端旋转) */
};

/* 
 * 【关节限位保护】
 * 逆解使用相对机械零位角度，不能在这里加入电机 Position 的 12.5rad 偏置 
 * 单位：弧度 (rad)。在逆解迭代中，如果计算出的角度超出这个范围，会被强制拉回（Clamp）。
 */
const float Arm_Joint_Min[ARM_CALCULATE_JOINT_COUNT] = {
  -3.0f, -3.0f, -1.5f, 0.0f, -1.5f, -0.25f, -1.5f
};

const float Arm_Joint_Max[ARM_CALCULATE_JOINT_COUNT] = {
  2.5f, 0.2f, 1.5f, 2.3f, 1.5f, 0.8f, 1.5f
};

/* ============================ 正运动学解算 (Forward Kinematics) ============================ */

/**
 * @brief  计算单个关节的 4x4 齐次变换矩阵 (Homogeneous Transformation Matrix)
 * @note   使用与参考工程一致的 改进型 DH 变换法则：
 *         T = RotX(alpha) * TransX(a) * RotZ(theta) * TransZ(d)
 * @param  matrix 输出的 4x4 矩阵，按行优先(一维数组16个元素)存储
 * @param  theta  关节角 (rad)
 * @param  alpha  扭转角 (rad)
 * @param  a      杆长 (m)
 * @param  d      偏置距 (m)
 * 
 * @usage  内部函数，输入 DH 结构体中的4个参数，得到此关节相对于上一个关节的位姿矩阵。
 */
void Arm_Calculate_DHMatrix(float matrix[16], float theta, float alpha, float a, float d)
{
  const float ct = cosf(theta);
  const float st = sinf(theta);
  const float ca = cosf(alpha);
  const float sa = sinf(alpha);

  /* 
   * 矩阵按行优先存储，前3x3为旋转矩阵 R，最后一列的前3个元素为平移向量 P
   * [ R11  R12  R13  Px ]
   * [ R21  R22  R23  Py ]
   * [ R31  R32  R33  Pz ]
   * [  0    0    0    1 ]
   */
  matrix[0] = ct;
  matrix[1] = -st;
  matrix[2] = 0.0f;
  matrix[3] = a;          /* Px */
  matrix[4] = st * ca;
  matrix[5] = ct * ca;
  matrix[6] = -sa;
  matrix[7] = -d * sa;    /* Py */
  matrix[8] = st * sa;
  matrix[9] = ct * sa;
  matrix[10] = ca;
  matrix[11] = d * ca;    /* Pz */
  matrix[12] = 0.0f;
  matrix[13] = 0.0f;
  matrix[14] = 0.0f;
  matrix[15] = 1.0f;
}

/**
 * @brief  4x4 矩阵乘法： result = left * right
 * @note   引入了中间缓存数组 product，这样即使用户传入的 result 指针与 left/right 相同，也不会导致数据覆盖污染。
 * @usage  用于不断累乘各个关节的 DH 矩阵。
 */
static void Arm_Calculate_MultiplyMatrix(const float left[16], const float right[16], float result[16])
{
  uint8_t row;
  uint8_t column;
  uint8_t index;
  float product[16];

  /* 使用临时矩阵，支持 result 与 left 或 right 指向同一存储区 */
  for (row = 0U; row < 4U; row++)
  {
    for (column = 0U; column < 4U; column++)
    {
      product[row * 4U + column] = 0.0f;
      for (index = 0U; index < 4U; index++)
        product[row * 4U + column] += left[row * 4U + index] * right[index * 4U + column];
    }
  }

  for (index = 0U; index < 16U; index++)
    result[index] = product[index];
}

/* Swap the public X and Z axes while preserving row-major homogeneous form.
 * Applying this map twice returns the original internal DH-frame transform. */
static void Arm_Calculate_SwapXZTransform(const float source[16], float destination[16])
{
  static const uint8_t axis_map[4] = {2U, 1U, 0U, 3U};
  uint8_t row;
  uint8_t column;

  for (row = 0U; row < 4U; row++)
  {
    for (column = 0U; column < 4U; column++)
      destination[row * 4U + column] = source[axis_map[row] * 4U + axis_map[column]];
  }
}

/**
 * @brief  计算正运动学 (Forward Kinematics, FK)
 * @param  joint_angle 输入的关节角数组（包含7个关节的当前角度，不含偏置）
 * @param  transform   输出的 4x4 末端齐次变换矩阵，表示夹爪相对于底座的 [位置+姿态]
 * 
 * @usage  当你获取到电机的真实角度后，调用此函数，就能知道机械臂末端在空间中的具体X,Y,Z坐标和旋转姿态。
 */
static void Arm_Calculate_ForwardKinematicsInternal(
    const float joint_angle[ARM_CALCULATE_JOINT_COUNT], float transform[16])
{
  uint8_t index;
  float current[16];
  float joint_transform[16];

  /* 从基座单位矩阵开始 (对角线为1，其余为0) */
  for (index = 0U; index < 16U; index++)
    current[index] = (index % 5U == 0U) ? 1.0f : 0.0f;

  /* 按 J1 到 J7 依次累乘，矩阵链乘：T_end = T_1 * T_2 * ... * T_7 */
  for (index = 0U; index < ARM_CALCULATE_JOINT_COUNT; index++)
  {
    Arm_Calculate_DHMatrix(joint_transform,
                           joint_angle[index] + Arm_DH_Table[index].theta_offset, /* 加上安装偏置 */
                           Arm_DH_Table[index].alpha,
                           Arm_DH_Table[index].a,
                           Arm_DH_Table[index].d);
    Arm_Calculate_MultiplyMatrix(current, joint_transform, current);
  }

  /* 输出最终末端矩阵 */
  for (index = 0U; index < 16U; index++)
    transform[index] = current[index];
}

void Arm_Calculate_ForwardKinematics(const float joint_angle[ARM_CALCULATE_JOINT_COUNT],
                                     float transform[16])
{
  float internal_transform[16];

  Arm_Calculate_ForwardKinematicsInternal(joint_angle, internal_transform);
  Arm_Calculate_SwapXZTransform(internal_transform, transform);
}

/* ============================ 逆运动学解算 (Inverse Kinematics) ============================ */

/* 计算三维向量叉积 (外积)：result = left x right */
static void Arm_Calculate_Vector3Cross(const float left[3], const float right[3], float result[3])
{
  result[0] = left[1] * right[2] - left[2] * right[1];
  result[1] = left[2] * right[0] - left[0] * right[2];
  result[2] = left[0] * right[1] - left[1] * right[0];
}

/* 计算三维向量长度 (L2范数，即欧氏距离) */
static float Arm_Calculate_Vector3Norm(const float vector[3])
{
  return sqrtf(vector[0] * vector[0] +
               vector[1] * vector[1] +
               vector[2] * vector[2]);
}

/**
 * @brief  计算雅可比矩阵 (Geometric Jacobian) 和当前的末端位姿
 * @note   雅可比矩阵 J 是一个 6行 x N列 的矩阵 (N=关节数=7)。
 *         它建立了“关节角速度”到“末端空间速度(线速度+角速度)”的线性映射关系： V_end = J * dq
 *         雅可比矩阵前三行为线速度部分，后三行为角速度部分。
 * 
 * @param  joint_angle 当前的关节角数组
 * @param  transform   输出的当前末端位姿矩阵 (附带计算正运动学，节省算力)
 * @param  jacobian    输出的 6x7 雅可比矩阵
 * 
 * @usage  内部函数，用于逆解迭代计算时的核心梯度求导。
 */
static void Arm_Calculate_Jacobian(
    const float joint_angle[ARM_CALCULATE_JOINT_COUNT],
    float transform[16],
    float jacobian[6][ARM_CALCULATE_JOINT_COUNT])
{
  uint8_t joint;
  uint8_t index;
  float current[16];
  float joint_transform[16];
  float joint_origin[ARM_CALCULATE_JOINT_COUNT][3]; // 保存每个关节的圆心空间坐标 P_i
  float joint_axis[ARM_CALCULATE_JOINT_COUNT][3];   // 保存每个关节的旋转轴向量 Z_i
  float end_position[3];
  float arm_vector[3];
  float linear_column[3];
  float sin_alpha;
  float cos_alpha;

  for (index = 0U; index < 16U; index++)
    current[index] = (index % 5U == 0U) ? 1.0f : 0.0f;

  for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
  {
    /* 
     * 【重点数学原理】
     * 改进 DH 中，关节 i 的旋转轴(Z轴)位于固定变换 RotX(alpha)*TransX(a) 之后。
     * 所以我们在乘入当前关节的 theta 之前，先提取当前坐标系下的原点和 Z 轴向量。
     */
    sin_alpha = sinf(Arm_DH_Table[joint].alpha);
    cos_alpha = cosf(Arm_DH_Table[joint].alpha);
    
    // 提取关节的原点坐标
    joint_origin[joint][0] = current[3] + Arm_DH_Table[joint].a * current[0];
    joint_origin[joint][1] = current[7] + Arm_DH_Table[joint].a * current[4];
    joint_origin[joint][2] = current[11] + Arm_DH_Table[joint].a * current[8];
    
    // 提取关节的 Z 轴方向向量 (角速度方向)
    joint_axis[joint][0] = -sin_alpha * current[1] + cos_alpha * current[2];
    joint_axis[joint][1] = -sin_alpha * current[5] + cos_alpha * current[6];
    joint_axis[joint][2] = -sin_alpha * current[9] + cos_alpha * current[10];

    // 继续累乘正运动学
    Arm_Calculate_DHMatrix(joint_transform,
                           joint_angle[joint] + Arm_DH_Table[joint].theta_offset,
                           Arm_DH_Table[joint].alpha,
                           Arm_DH_Table[joint].a,
                           Arm_DH_Table[joint].d);
    Arm_Calculate_MultiplyMatrix(current, joint_transform, current);
  }

  for (index = 0U; index < 16U; index++)
    transform[index] = current[index];

  // 提取最终末端(夹爪)的位置
  end_position[0] = current[3];
  end_position[1] = current[7];
  end_position[2] = current[11];

  /* 
   * 构造几何雅可比的列向量
   * J_v (线速度) = Z_i 叉乘 (P_end - P_i)
   * J_w (角速度) = Z_i
   */
  for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
  {
    arm_vector[0] = end_position[0] - joint_origin[joint][0];
    arm_vector[1] = end_position[1] - joint_origin[joint][1];
    arm_vector[2] = end_position[2] - joint_origin[joint][2];
    
    // 叉乘计算线速度部分
    Arm_Calculate_Vector3Cross(joint_axis[joint], arm_vector, linear_column);

    jacobian[0][joint] = linear_column[0];
    jacobian[1][joint] = linear_column[1];
    jacobian[2][joint] = linear_column[2];
    jacobian[3][joint] = joint_axis[joint][0];
    jacobian[4][joint] = joint_axis[joint][1];
    jacobian[5][joint] = joint_axis[joint][2];
  }
}

/**
 * @brief  计算当前位姿与目标位姿的 6D 空间误差 (3维位置误差 + 3维姿态误差)
 * @param  current           当前算出的末端 4x4 矩阵
 * @param  target            用户期望达到的目标 4x4 矩阵
 * @param  position_error    输出的 3D 位置误差 (X, Y, Z 的差值)
 * @param  orientation_error 输出的 3D 姿态误差 (基于轴角 Axis-Angle 表示法)
 * 
 * @usage  用于反馈给逆解迭代器，误差越小，说明越接近目标位姿。
 */
static void Arm_Calculate_PoseError(const float current[16],
                                    const float target[16],
                                    float position_error[3],
                                    float orientation_error[3])
{
  uint8_t row;
  uint8_t column;
  uint8_t index;
  float rotation_error[3][3];
  float cosine_angle;
  float angle;
  float sine_angle;
  float scale;
  float axis[3];
  float axis_square;
  float denominator;

  // 1. 位置误差直接相减： P_err = P_target - P_current
  position_error[0] = target[3] - current[3];
  position_error[1] = target[7] - current[7];
  position_error[2] = target[11] - current[11];

  /* 2. 姿态误差：计算相对旋转矩阵 R_error = R_target * R_current^T */
  /* 结果是在基座全局坐标系下表达的所需旋转量 */
  for (row = 0U; row < 3U; row++)
  {
    for (column = 0U; column < 3U; column++)
    {
      rotation_error[row][column] = 0.0f;
      for (index = 0U; index < 3U; index++)
        // 注意 current 矩阵提取的是转置项 current[column*4 + index]
        rotation_error[row][column] += target[row * 4U + index] * current[column * 4U + index];
    }
  }

  // 3. 从旋转矩阵中提取旋转角 (Angle)
  // 公式: tr(R) = 1 + 2*cos(theta)
  cosine_angle = 0.5f * (rotation_error[0][0] + rotation_error[1][1] +
                         rotation_error[2][2] - 1.0f);
  if (cosine_angle > 1.0f)
    cosine_angle = 1.0f;
  else if (cosine_angle < -1.0f)
    cosine_angle = -1.0f;
  angle = acosf(cosine_angle);

  // 4. 提取旋转轴 (Axis) 的非归一化向量，利用反对称特性
  orientation_error[0] = rotation_error[2][1] - rotation_error[1][2];
  orientation_error[1] = rotation_error[0][2] - rotation_error[2][0];
  orientation_error[2] = rotation_error[1][0] - rotation_error[0][1];

  // 5. 奇异点保护与归一化处理
  if (angle < 1.0e-5f)
  {
    // 如果角度极小，使用泰勒展开的一阶近似，避免除以零
    orientation_error[0] *= 0.5f;
    orientation_error[1] *= 0.5f;
    orientation_error[2] *= 0.5f;
  }
  else if (cosine_angle < -0.9999f)
  {
    /* 如果角度接近 180 度，常规的反对称提取法(sin)会变成 0，导致奇异。 
     * 此时需要从矩阵对角元素提取旋转轴。
     */
    axis_square = (rotation_error[0][0] + 1.0f) * 0.5f;
    axis[0] = sqrtf(axis_square > 0.0f ? axis_square : 0.0f);
    axis_square = (rotation_error[1][1] + 1.0f) * 0.5f;
    axis[1] = sqrtf(axis_square > 0.0f ? axis_square : 0.0f);
    axis_square = (rotation_error[2][2] + 1.0f) * 0.5f;
    axis[2] = sqrtf(axis_square > 0.0f ? axis_square : 0.0f);

    // 寻找最大分量，保证数值稳定性
    if ((axis[0] >= axis[1]) && (axis[0] >= axis[2]))
    {
      denominator = 4.0f * axis[0];
      if (denominator > 1.0e-6f)
      {
        axis[1] = (rotation_error[0][1] + rotation_error[1][0]) / denominator;
        axis[2] = (rotation_error[0][2] + rotation_error[2][0]) / denominator;
      }
    }
    else if (axis[1] >= axis[2])
    {
      denominator = 4.0f * axis[1];
      if (denominator > 1.0e-6f)
      {
        axis[0] = (rotation_error[0][1] + rotation_error[1][0]) / denominator;
        axis[2] = (rotation_error[1][2] + rotation_error[2][1]) / denominator;
      }
    }
    else
    {
      denominator = 4.0f * axis[2];
      if (denominator > 1.0e-6f)
      {
        axis[0] = (rotation_error[0][2] + rotation_error[2][0]) / denominator;
        axis[1] = (rotation_error[1][2] + rotation_error[2][1]) / denominator;
      }
    }

    // 将轴乘以角度，获得最终的姿态误差向量 (Axis-Angle)
    orientation_error[0] = angle * axis[0];
    orientation_error[1] = angle * axis[1];
    orientation_error[2] = angle * axis[2];
  }
  else
  {
    // 常规情况：直接除以 2*sin(angle) 进行归一化，并乘以 angle
    sine_angle = sinf(angle);
    scale = angle / (2.0f * sine_angle);
    orientation_error[0] *= scale;
    orientation_error[1] *= scale;
    orientation_error[2] *= scale;
  }
}

/**
 * @brief  使用带“列主元”的高斯消元法求解 6x6 的线性方程组： A * x = b
 * @param  matrix   系数矩阵 A (6x6)
 * @param  vector   常数项向量 b (6x1)
 * @param  solution 输出的解向量 x (6x1)
 * @return 1=成功，0=矩阵奇异(无解)
 * 
 * @usage  逆向运动学迭代中，用于求解 (J*J^T + λ^2*I) * y = error 线性方程。
 */
static uint8_t Arm_Calculate_Solve6x6(float matrix[6][6],
                                      const float vector[6],
                                      float solution[6])
{
  uint8_t column;
  uint8_t row;
  uint8_t pivot_row;
  uint8_t index;
  float augmented[6][7]; // 增广矩阵 (6行7列)
  float pivot_abs;
  float candidate_abs;
  float factor;
  float temp;

  // 1. 构造增广矩阵 [A | b]
  for (row = 0U; row < 6U; row++)
  {
    for (column = 0U; column < 6U; column++)
      augmented[row][column] = matrix[row][column];
    augmented[row][6] = vector[row];
  }

  // 2. 高斯消元 (化为上三角矩阵)
  for (column = 0U; column < 6U; column++)
  {
    // 选列主元：找出当前列中绝对值最大的元素所在行，防止浮点数精度丢失
    pivot_row = column;
    pivot_abs = fabsf(augmented[pivot_row][column]);
    for (row = (uint8_t)(column + 1U); row < 6U; row++)
    {
      candidate_abs = fabsf(augmented[row][column]);
      if (candidate_abs > pivot_abs)
      {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }

    // 如果主元接近0，说明矩阵不可逆（奇异）
    if (pivot_abs < 1.0e-9f)
      return 0U;

    // 交换行，将主元换到对角线上
    if (pivot_row != column)
    {
      for (index = column; index < 7U; index++)
      {
        temp = augmented[column][index];
        augmented[column][index] = augmented[pivot_row][index];
        augmented[pivot_row][index] = temp;
      }
    }

    // 消去当前列下方的所有元素
    for (row = (uint8_t)(column + 1U); row < 6U; row++)
    {
      factor = augmented[row][column] / augmented[column][column];
      for (index = column; index < 7U; index++)
        augmented[row][index] -= factor * augmented[column][index];
    }
  }

  // 3. 回代求解 (Back Substitution)
  for (row = 6U; row > 0U; row--)
  {
    index = (uint8_t)(row - 1U);
    temp = augmented[index][6];
    for (column = (uint8_t)(index + 1U); column < 6U; column++)
      temp -= augmented[index][column] * solution[column];
    solution[index] = temp / augmented[index][index];
  }

  return 1U;
}

/**
 * @brief  获取逆向运动学的默认配置参数
 * @param  config 待赋值的配置结构体
 * 
 * @usage  如果你不知道怎么调参，就直接调用此函数获取稳健的默认参数传入 IK 函数。
 */
void Arm_Calculate_GetDefaultIKConfig(Arm_IK_Config_t *config)
{
  if (config == 0)
    return;

  config->max_iterations = 150U;        // 最大迭代次数，超出会报不收敛
  config->position_tolerance = 0.001f;  // 位置容差：1mm
  config->orientation_tolerance = 0.01f;// 姿态容差：0.01rad (约0.57度)
  config->damping = 0.03f;              // 阻尼因子 lambda，用于处理奇异点，越大约不准但越稳
  config->max_joint_step = 0.12f;       // 单次迭代中，单个关节允许改变的最大步长(防止跳跃)
  config->orientation_weight = 0.20f;   // 姿态权重 (相对于位置)，可以根据需要着重位置还是姿态
}

/**
 * @brief  计算逆运动学 (Inverse Kinematics, IK) - 核心函数
 * @note   该函数使用 **阻尼最小二乘法 (Damped Least Squares, DLS / Levenberg-Marquardt)**。
 *         这是处理 7自由度冗余机械臂 的最佳方案，不仅能应对多解，还能在奇异点(死角)处平滑过度。
 * 
 * @param  target_transform    [入参] 你期望机械臂末端达到的 4x4 目标位姿矩阵。
 * @param  initial_joint_angle [入参] 迭代初始猜测值，通常传入机械臂*当前*的真实关节角。
 * @param  joint_angle_out     [出参] 计算得出的 7 个关节目标角度。
 * @param  config              [入参] 配置参数，传 NULL 则自动使用默认参数。
 * 
 * @retval ARM_IK_SUCCESS        成功收敛，找到了解。
 * @retval ARM_IK_NOT_CONVERGED  达到最大迭代次数仍未达到容差要求（距离太远或无解）。
 * @retval ARM_IK_SINGULAR       矩阵奇异且阻尼配置失效（极少发生）。
 * 
 * @usage  当你想让夹爪去抓一个空间坐标 (X,Y,Z, Rx,Ry,Rz) 时，把这个坐标转化为 target_transform 矩阵，
 *         调用此函数，就能算出 7 个电机的目标角度，然后发给底层的 PID 即可控制机械臂。
 */
Arm_IK_Status_t Arm_Calculate_InverseKinematics(
    const float target_transform[16],
    const float initial_joint_angle[ARM_CALCULATE_JOINT_COUNT],
    float joint_angle_out[ARM_CALCULATE_JOINT_COUNT],
    const Arm_IK_Config_t *config)
{
  uint16_t iteration;
  uint8_t row;
  uint8_t column;
  uint8_t joint;
  Arm_IK_Config_t default_config;
  const Arm_IK_Config_t *active_config = config;
  float current_transform[16];
  float internal_target_transform[16];
  float jacobian[6][ARM_CALCULATE_JOINT_COUNT];
  float normal_matrix[6][6];
  float pose_error[6];
  float position_error[3];
  float orientation_error[3];
  float solved_error[6] = {0.0f};
  float working_joint[ARM_CALCULATE_JOINT_COUNT];
  float joint_step;
  float damping_square;

  if ((target_transform == 0) || (initial_joint_angle == 0) || (joint_angle_out == 0))
    return ARM_IK_INVALID_ARGUMENT;

  /* IK iterates in the internal DH frame; convert the public X/Z-swapped
     target pose back to that frame first. */
  Arm_Calculate_SwapXZTransform(target_transform, internal_target_transform);

  /* 
   * 先把输出值填为初始角。
   * 这样做的目的是：确保任何失败状态(比如算不出解)都不会向调用者留下脏数据，
   * 导致机械臂猛烈抽搐跳动。如果失败了，机械臂就待在原地不动。
   */
  for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
    joint_angle_out[joint] = initial_joint_angle[joint];

  if (active_config == 0)
  {
    Arm_Calculate_GetDefaultIKConfig(&default_config);
    active_config = &default_config;
  }

  // 参数安全检查
  if ((active_config->max_iterations == 0U) ||
      (active_config->position_tolerance <= 0.0f) ||
      (active_config->orientation_tolerance <= 0.0f) ||
      (active_config->damping <= 0.0f) ||
      (active_config->max_joint_step <= 0.0f) ||
      (active_config->orientation_weight <= 0.0f))
    return ARM_IK_INVALID_ARGUMENT;

  // 将工作变量初始化为传入的初始角度，并进行第一次关节限位 Clamp
  for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
  {
    working_joint[joint] = initial_joint_angle[joint];
    if (working_joint[joint] < Arm_Joint_Min[joint])
      working_joint[joint] = Arm_Joint_Min[joint];
    else if (working_joint[joint] > Arm_Joint_Max[joint])
      working_joint[joint] = Arm_Joint_Max[joint];
  }

  damping_square = active_config->damping * active_config->damping; // 计算 λ^2

  /* ---------------- 开始迭代求解 ---------------- */
  for (iteration = 0U; iteration < active_config->max_iterations; iteration++)
  {
    // 1. 计算当前的雅可比矩阵 J 和当前的位姿 T_current
    Arm_Calculate_Jacobian(working_joint, current_transform, jacobian);
    
    // 2. 根据 T_current 和 T_target 计算 6D 空间误差 e
    Arm_Calculate_PoseError(current_transform, internal_target_transform,
                            position_error, orientation_error);

    // 3. 收敛判断：如果位置和姿态误差都小于设定的容差，视为成功！
    if ((Arm_Calculate_Vector3Norm(position_error) <= active_config->position_tolerance) &&
        (Arm_Calculate_Vector3Norm(orientation_error) <= active_config->orientation_tolerance))
    {
      for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
        joint_angle_out[joint] = working_joint[joint]; // 保存解
      return ARM_IK_SUCCESS;
    }

    // 4. 将误差打包成一个 6x1 的向量，并给姿态误差施加权重
    pose_error[0] = position_error[0];
    pose_error[1] = position_error[1];
    pose_error[2] = position_error[2];
    pose_error[3] = orientation_error[0] * active_config->orientation_weight;
    pose_error[4] = orientation_error[1] * active_config->orientation_weight;
    pose_error[5] = orientation_error[2] * active_config->orientation_weight;

    /* 同理，对雅可比矩阵的角速度部分施加相同的姿态权重，使得方程单位统一 */
    for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
    {
      jacobian[3][joint] *= active_config->orientation_weight;
      jacobian[4][joint] *= active_config->orientation_weight;
      jacobian[5][joint] *= active_config->orientation_weight;
    }

    /* 
     * 5. 阻尼最小二乘核心数学：
     * 我们需要求解的更新步长 dq 的公式为：
     *     dq = J^T * (J * J^T + λ^2 * I)^-1 * e
     * 
     * 令 y = (J * J^T + λ^2 * I)^-1 * e
     * 即可将问题转化为解线性方程组： (J * J^T + λ^2 * I) * y = e
     * 
     * 首先，构造 Normal Matrix = (J * J^T + λ^2 * I)
     */
    for (row = 0U; row < 6U; row++)
    {
      for (column = 0U; column < 6U; column++)
      {
        normal_matrix[row][column] = 0.0f;
        for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
          normal_matrix[row][column] += jacobian[row][joint] * jacobian[column][joint]; // J * J^T
      }
      normal_matrix[row][row] += damping_square; // 加上 λ^2 * I
    }

    // 6. 使用高斯消元法解出中间变量 y (代码中的 solved_error)
    if (Arm_Calculate_Solve6x6(normal_matrix, pose_error, solved_error) == 0U)
      return ARM_IK_SINGULAR; // 方程无解

    /* 
     * 7. 计算最终的关节更新步长 dq = J^T * y
     * 并将其累加到当前关节角上。
     */
    for (joint = 0U; joint < ARM_CALCULATE_JOINT_COUNT; joint++)
    {
      joint_step = 0.0f;
      for (row = 0U; row < 6U; row++)
        // 注意这里是 J^T，所以使用 jacobian[row][joint] (即矩阵转置相乘)
        joint_step += jacobian[row][joint] * solved_error[row];

      // 步长限制 (防止在奇点附近发散导致机械臂乱转)
      if (joint_step > active_config->max_joint_step)
        joint_step = active_config->max_joint_step;
      else if (joint_step < -active_config->max_joint_step)
        joint_step = -active_config->max_joint_step;

      // 更新角度
      working_joint[joint] += joint_step;
      
      // 8. 关节限位保护 (Clamp)，不允许越界
      if (working_joint[joint] < Arm_Joint_Min[joint])
        working_joint[joint] = Arm_Joint_Min[joint];
      else if (working_joint[joint] > Arm_Joint_Max[joint])
        working_joint[joint] = Arm_Joint_Max[joint];
    }
  }

  // 如果循环结束还没有 return ARM_IK_SUCCESS，说明没收敛
  return ARM_IK_NOT_CONVERGED;
}

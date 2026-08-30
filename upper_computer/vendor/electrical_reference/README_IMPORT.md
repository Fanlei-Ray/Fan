# 电控截图导入说明

导入日期：2026-08-26。原始截图由电控同学通过课程项目群提供，项目内副本仅用于接口核对，不作为电机写控制授权。

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `motor_j1_can_tool.png` | CAN ID 1、MST ID 显示 11、J1 工具 P/V/T 范围 | `D66BC18F37869505D01CF01AFE705B32373CDA74E4ABAE2622356BA14EDCF7E5` |
| `motor_j2_driver_params.png` | CAN ID 0x2、Master ID 0x12、CAN 1M、驱动参数 | `EE8EE54C7FB081C29AFF392C59FBF1C377114C443D9448E916D4D9FF346AA446` |
| `motor_j4_can_tool.png` | CAN ID 4、MST ID 显示 14、J4 工具 P/V/T 范围 | `9ACB2057CBBEBB629EC1126DDDFA756751BE9AF4173C03B92E954BE88313C550` |
| `motor_j5_can_tool.png` | CAN ID 5、MST ID 显示 15、J5 工具 P/V/T 范围 | `A091E86F0C0F10CB8F208A4640D3E32E2DDCCC0E1410AA709877EEBDFC0D6B14` |

截图里 P/V/T 的上下限是电机调试工具显示范围，不等同于整机关节批准的安全限值。减速比字段虽然在 J2 参数截图中出现，但电控同学明确要求上位机不处理减速比。

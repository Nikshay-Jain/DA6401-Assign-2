# DA6401-Assign-2
## Nikshay Jain | MM21B044

```bash
# Run Part A with hyperparameter sweep
!python main.py --part a --sweep --data_dir {DATA_DIR}

# Train the final model for Part A
!python main.py --part a --train --test --data_dir {DATA_DIR}

# Run Part B (fine-tuning)
!python main.py --part b --train --test --data_dir {DATA_DIR}
```
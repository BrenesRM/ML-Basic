import random
import matplotlib.pyplot as plt

# Simple Homomorphic Encryption Scheme (Additive)
def encrypt(value, key):
    """Encrypts a value using a key."""
    return value + key

def decrypt(encrypted_value, key):
    """Decrypts a value using a key."""
    return encrypted_value - key

def add_encrypted_values(enc_val1, enc_val2):
    """Adds two encrypted values."""
    return enc_val1 + enc_val2

# Example Usage
if __name__ == "__main__":
    # Original data
    original_values = [random.randint(1, 10) for _ in range(5)]

    # Encryption key (randomly chosen)
    encryption_key = random.randint(10, 20)

    # Encrypt the values
    encrypted_values = [encrypt(val, encryption_key) for val in original_values]

    # Perform computation on encrypted values (e.g., summation)
    encrypted_sum = add_encrypted_values(encrypted_values[0], encrypted_values[1])

    # Decrypt the result
    decrypted_sum = decrypt(encrypted_sum, encryption_key)

    # Visual Representation
    plt.figure(figsize=(10, 6))

    # Original Values
    plt.plot(original_values, label="Original Values", marker="o")

    # Encrypted Values
    plt.plot(encrypted_values, label="Encrypted Values", marker="o")

    # Decrypted Computation Result
    plt.axhline(decrypted_sum, color="red", linestyle="--", label=f"Decrypted Sum: {decrypted_sum}")

    plt.title("Homomorphic Encryption Example")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()

    # Output Results
    print("Original Values:", original_values)
    print("Encrypted Values:", encrypted_values)
    print("Decrypted Sum of First Two Values:", decrypted_sum)
